import csv
import gc
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import optim
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

#from pre_data import read_data

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_MODEL_DIRS = [
    CURRENT_DIR,
    os.path.join(os.path.dirname(CURRENT_DIR), "model"),
    "",
    "",
]
for path in CANDIDATE_MODEL_DIRS:
    if os.path.exists(os.path.join(path, "pre_data.py")) and path not in sys.path:
        sys.path.append(path)
from pre_data import read_data


CONFIG = {
    "device_id": "0",
    "model_path": "pre_model/DeepSeek-Coder-1.3b-instruct",
    "data_path": "data/LM-LPL.json",
    "log_path": "results/DeepSeek_finetune.log",
    "save_dir": "save_dict",
    "metrics_csv": "results/DeepSeek_finetune_metrics.csv",
    # Keep these hyperparameters aligned with the prompt-learning scripts.
    "batch_size": 4,
    "epochs": 8,
    "iters": 3,
    "gradient_accumulation_steps": 2,
    "seed": 42,
    "max_seq_length": 512,
    "lora_rank": 16,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "lora_dropout": 0.05,
    "num_labels": 4,
    "early_stopping_patience": 3,
    "min_delta": 1e-4,
}

os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["device_id"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ModelComponents:
    model: torch.nn.Module
    tokenizer: AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging() -> None:
    os.makedirs(os.path.dirname(CONFIG["log_path"]), exist_ok=True)
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(CONFIG["log_path"], mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_deepseek_finetune(model_path: str) -> ModelComponents:
    logging.info(f"Loading model for code-only SeqCls fine-tuning: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=CONFIG["num_labels"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_rank"] * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Baseline type: code-only sequence-classification fine-tuning.")
    logging.info("Input contains raw Java method code only: no prompt, no label description, no examples.")

    return ModelComponents(model=model, tokenizer=tokenizer)


def create_dataloaders(train_data, val_data, test_data, tokenizer: AutoTokenizer):
    def build_loader(data, is_train: bool = False):
        input_ids_list = []
        attention_mask_list = []
        label_list = []

        for item in data:
            encoding = tokenizer(
                item.text_a,
                add_special_tokens=True,
                truncation=True,
                max_length=CONFIG["max_seq_length"],
                padding="max_length",
                return_tensors="pt",
            )
            input_ids_list.append(encoding["input_ids"].squeeze(0))
            attention_mask_list.append(encoding["attention_mask"].squeeze(0))
            label_list.append(int(item.label))

        dataset = torch.utils.data.TensorDataset(
            torch.stack(input_ids_list),
            torch.stack(attention_mask_list),
            torch.tensor(label_list, dtype=torch.long),
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=is_train,
            num_workers=4,
            pin_memory=True,
        )

    return (
        build_loader(train_data, is_train=True),
        build_loader(val_data, is_train=False),
        build_loader(test_data, is_train=False),
    )


def evaluate_model(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    best_val_f1: float = -1,
    best_model_path: str = "",
    stage: str = "eval",
) -> Tuple[float, float, float, float, float]:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

    model.eval()
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for batch in data_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    acc = accuracy_score(y_true, y_pred)
    pre, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, class_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(CONFIG["num_labels"])),
        average=None,
        zero_division=0,
    )
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    logging.info("\n--- evaluation metrics ---")
    logging.info(f"Accuracy             : {acc:.4f}")
    logging.info(f"Weighted Precision   : {pre:.4f}")
    logging.info(f"Weighted Recall      : {rec:.4f}")
    logging.info(f"Weighted F1-score    : {f1:.4f}")
    logging.info(f"Macro F1-score       : {macro_f1:.4f}")
    logging.info(f"Class-3 F1-score     : {class_f1[3]:.4f}")
    logging.info(f"\n--- classification report ---\n{report}")
    logging.info(f"\n--- confusion matrix ---\n{cm}")

    if stage == "eval" and f1 > best_val_f1:
        best_val_f1 = f1
        torch.save(model.state_dict(), best_model_path)
        logging.info(f"New best model saved with F1: {f1:.4f}")

    return acc, pre, rec, f1, best_val_f1


def train_model(
    components: ModelComponents,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    save_model_path: str,
):
    model = components.model
    model.to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        foreach=True,
    )
    num_training_steps = len(train_loader) * CONFIG["epochs"] // CONFIG["gradient_accumulation_steps"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    best_val_f1 = -1
    bad_epochs = 0
    optimizer.zero_grad()

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}")):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / CONFIG["gradient_accumulation_steps"]

            loss.backward()

            if (step + 1) % CONFIG["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * CONFIG["gradient_accumulation_steps"]

        logging.info(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")
        logging.info("\n--- eval ---")
        previous_best = best_val_f1
        _, _, _, _, _, _, best_val_f1 = evaluate_model(
            model,
            val_loader,
            device,
            best_val_f1=best_val_f1,
            best_model_path=save_model_path,
            stage="eval",
        )

        if best_val_f1 > previous_best + CONFIG["min_delta"]:
            bad_epochs = 0
        else:
            bad_epochs += 1
            logging.info(f"Early stopping counter: {bad_epochs}/{CONFIG['early_stopping_patience']}")
            if bad_epochs >= CONFIG["early_stopping_patience"]:
                logging.info("Early stopping triggered.")
                break

    del optimizer
    del scheduler
    torch.cuda.empty_cache()
    gc.collect()

    logging.info("\n--- testing ---")
    model.load_state_dict(torch.load(save_model_path, map_location="cpu"))
    return evaluate_model(model, test_loader, device, stage="testing")[:4]


def save_metrics(acc: float, pre: float, rec: float, f1: float, csv_path: str) -> None:
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Accuracy", "Precision", "Recall", "F1"])
        writer.writerow([round(acc, 4), round(pre, 4), round(rec, 4), round(f1, 4)])


def main() -> None:
    setup_logging()
    logging.info(f"Run name: DeepSeek_finetune_earlystop")
    logging.info(f"Model path: {CONFIG['model_path']}")
    logging.info(f"Data path: {CONFIG['data_path']}")
    logging.info(f"Log path: {CONFIG['log_path']}")
    logging.info(f"Batch size: {CONFIG['batch_size']}")
    logging.info(f"Epochs: {CONFIG['epochs']}")
    logging.info(f"Max seq length: {CONFIG['max_seq_length']}")
    logging.info(f"LoRA rank: {CONFIG['lora_rank']}")
    logging.info(f"Learning rate: {CONFIG['learning_rate']}")
    logging.info(f"Early stopping patience: {CONFIG['early_stopping_patience']}")
    logging.info("No prompt text, no label mapping text, no examples, no handcrafted features.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    if device.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    train_data, val_data, test_data = read_data(CONFIG["data_path"])

    total_acc, total_pre, total_rec, total_f1 = [], [], [], []

    for iter_num in range(CONFIG["iters"]):
        set_seed(CONFIG["seed"] + iter_num)
        save_model_path = os.path.join(CONFIG["save_dir"], f"DeepSeek_finetune_earlystop_iter{iter_num + 1}_best.pth")

        components = load_deepseek_finetune(CONFIG["model_path"])
        train_loader, val_loader, test_loader = create_dataloaders(
            train_data,
            val_data,
            test_data,
            components.tokenizer,
        )

        acc, pre, rec, f1 = train_model(
            components,
            train_loader,
            val_loader,
            test_loader,
            device,
            save_model_path,
        )

        total_acc.append(acc)
        total_pre.append(pre)
        total_rec.append(rec)
        total_f1.append(f1)
        save_metrics(acc, pre, rec, f1, CONFIG["metrics_csv"])

        del components, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    logging.info("\n==================== final results ====================")
    logging.info(f"Accuracy           : {np.mean(total_acc):.4f} +/- {np.std(total_acc):.4f}")
    logging.info(f"Weighted Precision : {np.mean(total_pre):.4f} +/- {np.std(total_pre):.4f}")
    logging.info(f"Weighted Recall    : {np.mean(total_rec):.4f} +/- {np.std(total_rec):.4f}")
    logging.info(f"Weighted F1-score  : {np.mean(total_f1):.4f} +/- {np.std(total_f1):.4f}")


if __name__ == "__main__":
    main()


