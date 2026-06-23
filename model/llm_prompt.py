import argparse
import csv
import gc
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch import optim
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

from pre_data import read_data


CONFIG = {
    "device_id": "0",
    # Default DeepSeek setting. These paths and hyperparameters are aligned
    # with the existing decoder comparison scripts for fair comparison.
    "model_path": "pre_model/DeepSeek-Coder-1.3b-instruct",
    "data_path": "data/LM-LPL.json",
    "output_dir": "results/decoder_prompt_v3_light",
    "run_name": "DeepSeek_prompt_v3_light",
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
    "balanced_sampling": False,
    "score_normalization": "mean",
    "use_contextual_calibration": False,
    "calibration_alpha": 0.0,
    "selection_metric": "weighted_f1",
    "answer_mode": "semantic",
    "early_stopping_patience": 3,
    "min_delta": 1e-4,
    "use_few_shot": False,
}

NUM_LABELS = 4

# Training targets. Keep them semantic and boundary-safe: no class phrase should
# be a prefix of another class phrase after tokenization.
CANONICAL_ANSWERS: Dict[int, str] = {
    0: "no smell.",
    1: "long parameter list.",
    2: "long method.",
    3: "both smells.",
}

NUMERIC_ANSWERS: Dict[int, str] = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
}

# Inference verbalizer. The class score is the best normalized likelihood among
# its candidate phrases.
VERBALIZER: Dict[int, List[str]] = {
    0: ["no smell."],
    1: ["long parameter list."],
    2: ["long method."],
    3: ["both smells."],
}

NUMERIC_VERBALIZER: Dict[int, List[str]] = {
    0: ["0"],
    1: ["1"],
    2: ["2"],
    3: ["3"],
}

SYSTEM_PROMPT = (
    "You are a software quality expert. Detect only Long Method and Long Parameter List "
    "code smells in Java methods. Return only the requested label, with no explanation."
)

TASK_INSTRUCTION = (
    "Java method:\n"
    "{code}\n\n"
    "Code smell label: "
)

NUMERIC_TASK_INSTRUCTION = (
    "Java method:\n"
    "{code}\n\n"
    "Code smell label number: "
)

FEW_SHOT_BLOCK = (
    "Examples:\n"
    "Java method:\n"
    "```java\n"
    "int add(int a, int b) { return a + b; }\n"
    "```\n"
    "Answer: no smell.\n\n"
    "Java method:\n"
    "```java\n"
    "void configure(String a, String b, String c, String d, String e, String f) { }\n"
    "```\n"
    "Answer: long parameter list.\n\n"
)

CALIBRATION_CODE = "void placeholder() {\n}\n"


@dataclass
class ModelComponents:
    model: torch.nn.Module
    tokenizer: AutoTokenizer
    calibration_scores: torch.Tensor | None = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=CONFIG["model_path"])
    parser.add_argument("--data_path", default=CONFIG["data_path"])
    parser.add_argument("--output_dir", default=CONFIG["output_dir"])
    parser.add_argument("--run_name", default=CONFIG["run_name"])
    parser.add_argument("--device_id", default=CONFIG["device_id"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--iters", type=int, default=CONFIG["iters"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--max_seq_length", type=int, default=CONFIG["max_seq_length"])
    parser.add_argument("--learning_rate", type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--lora_rank", type=int, default=CONFIG["lora_rank"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=CONFIG["gradient_accumulation_steps"])
    parser.add_argument("--balanced_sampling", action="store_true")
    parser.add_argument("--enable_contextual_calibration", action="store_true")
    parser.add_argument("--disable_contextual_calibration", action="store_true")
    parser.add_argument("--calibration_alpha", type=float, default=CONFIG["calibration_alpha"])
    parser.add_argument("--answer_mode", choices=["semantic", "numeric"], default=CONFIG["answer_mode"])
    parser.add_argument("--early_stopping_patience", type=int, default=CONFIG["early_stopping_patience"])
    parser.add_argument("--min_delta", type=float, default=CONFIG["min_delta"])
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument(
        "--selection_metric",
        choices=["weighted_f1", "weighted_macro_class3"],
        default=CONFIG["selection_metric"],
    )
    return parser.parse_args()


def apply_args(args) -> None:
    CONFIG["model_path"] = args.model_path
    CONFIG["data_path"] = args.data_path
    CONFIG["output_dir"] = args.output_dir
    CONFIG["run_name"] = args.run_name
    CONFIG["device_id"] = args.device_id
    CONFIG["epochs"] = args.epochs
    CONFIG["iters"] = args.iters
    CONFIG["batch_size"] = args.batch_size
    CONFIG["max_seq_length"] = args.max_seq_length
    CONFIG["learning_rate"] = args.learning_rate
    CONFIG["lora_rank"] = args.lora_rank
    CONFIG["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    CONFIG["balanced_sampling"] = args.balanced_sampling
    CONFIG["use_contextual_calibration"] = args.enable_contextual_calibration and not args.disable_contextual_calibration
    CONFIG["calibration_alpha"] = args.calibration_alpha
    CONFIG["selection_metric"] = args.selection_metric
    CONFIG["answer_mode"] = args.answer_mode
    CONFIG["early_stopping_patience"] = args.early_stopping_patience
    CONFIG["min_delta"] = args.min_delta
    CONFIG["use_few_shot"] = args.use_few_shot
    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["device_id"]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging() -> Tuple[str, str]:
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    log_path = os.path.join(CONFIG["output_dir"], f"{CONFIG['run_name']}.log")
    metrics_csv = os.path.join(CONFIG["output_dir"], f"{CONFIG['run_name']}_metrics.csv")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path, metrics_csv


def load_model(model_path: str) -> ModelComponents:
    logging.info(f"Loading CausalLM for semantic prompt learning: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_rank"] * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_verbalizer_tokens(tokenizer)
    return ModelComponents(model=model, tokenizer=tokenizer)


def log_verbalizer_tokens(tokenizer: AutoTokenizer) -> None:
    logging.info("Semantic verbalizer tokenization:")
    for label_id in range(NUM_LABELS):
        for phrase in get_verbalizer()[label_id]:
            token_ids = tokenizer.encode(phrase, add_special_tokens=False)
            logging.info(f"  class {label_id}: {phrase!r} -> len={len(token_ids)}, ids={token_ids}")


def get_answers() -> Dict[int, str]:
    return NUMERIC_ANSWERS if CONFIG["answer_mode"] == "numeric" else CANONICAL_ANSWERS


def get_verbalizer() -> Dict[int, List[str]]:
    return NUMERIC_VERBALIZER if CONFIG["answer_mode"] == "numeric" else VERBALIZER


def encode_target_after_prompt(tokenizer: AutoTokenizer, prompt: str, target: str) -> Tuple[List[int], List[int]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    full_ids = tokenizer.encode(prompt + target, add_special_tokens=True)
    if len(full_ids) >= len(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids:
        return prompt_ids, full_ids[len(prompt_ids):]
    return prompt_ids, tokenizer.encode(target, add_special_tokens=False)


def truncate_code(tokenizer: AutoTokenizer, code: str, max_code_tokens: int) -> str:
    code_tokens = tokenizer.tokenize(code)
    if len(code_tokens) <= max_code_tokens:
        return code

    marker = "\n//[Truncated]"
    marker_tokens = tokenizer.tokenize(marker)
    keep_len = max(1, max_code_tokens - len(marker_tokens))
    kept_tokens = code_tokens[:keep_len] + marker_tokens
    return tokenizer.convert_tokens_to_string(kept_tokens)


def build_user_content(code: str) -> str:
    instruction = NUMERIC_TASK_INSTRUCTION if CONFIG["answer_mode"] == "numeric" else TASK_INSTRUCTION
    if CONFIG["use_few_shot"]:
        instruction = FEW_SHOT_BLOCK + instruction
    return instruction.format(code=code)


def build_prompt(tokenizer: AutoTokenizer, code: str, max_code_tokens: int) -> str:
    code = truncate_code(tokenizer, code, max_code_tokens)
    return build_user_content(code)


def get_max_code_tokens(tokenizer: AutoTokenizer) -> int:
    skeleton = build_prompt(tokenizer, "", 1)
    skeleton_len = len(tokenizer.tokenize(skeleton))
    longest_answer_len = max(len(tokenizer.encode(v, add_special_tokens=False)) for v in get_answers().values())
    max_code_tokens = CONFIG["max_seq_length"] - skeleton_len - longest_answer_len - 10
    if max_code_tokens <= 32:
        raise ValueError("max_seq_length is too small for this prompt template.")
    return max_code_tokens


def create_dataloaders(train_data, val_data, test_data, tokenizer: AutoTokenizer):
    max_code_tokens = get_max_code_tokens(tokenizer)
    logging.info(f"Max code tokens reserved by prompt template: {max_code_tokens}")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def build_loader(data, is_train: bool = False):
        input_ids_list: List[List[int]] = []
        labels_list: List[Sequence[int]] = []
        class_labels: List[int] = []

        for item in data:
            label = int(item.label)
            prompt = build_prompt(tokenizer, item.text_a, max_code_tokens)
            if is_train:
                prompt_ids, target_ids = encode_target_after_prompt(tokenizer, prompt, get_answers()[label])
                ids = prompt_ids + target_ids
                lbls = [-100] * len(prompt_ids) + target_ids
            else:
                ids = tokenizer.encode(prompt, add_special_tokens=True)
                lbls = label

            if len(ids) > CONFIG["max_seq_length"]:
                ids = ids[-CONFIG["max_seq_length"]:]
                if is_train:
                    lbls = list(lbls)[-CONFIG["max_seq_length"]:]

            input_ids_list.append(ids)
            labels_list.append(lbls)
            class_labels.append(label)

        max_len = min(max(len(ids) for ids in input_ids_list), CONFIG["max_seq_length"])
        padded_input_ids, padded_attention_mask, padded_labels = [], [], []

        for ids, lbls in zip(input_ids_list, labels_list):
            pad_len = max_len - len(ids)
            padded_input_ids.append([pad_id] * pad_len + ids)
            padded_attention_mask.append([0] * pad_len + [1] * len(ids))
            if is_train:
                padded_labels.append([-100] * pad_len + list(lbls))
            else:
                padded_labels.append(lbls)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(padded_input_ids, dtype=torch.long),
            torch.tensor(padded_attention_mask, dtype=torch.long),
            torch.tensor(padded_labels, dtype=torch.long),
        )

        sampler = None
        shuffle = is_train
        if is_train and CONFIG["balanced_sampling"]:
            counts = np.bincount(np.array(class_labels), minlength=NUM_LABELS)
            weights = [1.0 / max(counts[label], 1) for label in class_labels]
            sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            shuffle = False

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=shuffle,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )

    return build_loader(train_data, True), build_loader(val_data, False), build_loader(test_data, False)


def score_candidate_batch(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt_ids_batch: List[List[int]],
    candidate_text: str,
    device: torch.device,
) -> torch.Tensor:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    candidate_ids = tokenizer.encode(candidate_text, add_special_tokens=False)
    sequences, target_masks = [], []

    for prompt_ids in prompt_ids_batch:
        seq = prompt_ids + candidate_ids
        mask = [0] * len(prompt_ids) + [1] * len(candidate_ids)
        sequences.append(seq)
        target_masks.append(mask)

    max_len = max(len(seq) for seq in sequences)
    input_ids, attention_mask, target_mask = [], [], []
    for seq, mask in zip(sequences, target_masks):
        pad_len = max_len - len(seq)
        input_ids.append([pad_id] * pad_len + seq)
        attention_mask.append([0] * pad_len + [1] * len(seq))
        target_mask.append([0] * pad_len + mask)

    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long, device=device)
    target_mask_tensor = torch.tensor(target_mask, dtype=torch.bool, device=device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids_tensor[:, 1:]
    shift_target_mask = target_mask_tensor[:, 1:]
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_scores = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    summed = (token_scores * shift_target_mask.float()).sum(dim=-1)
    lengths = shift_target_mask.sum(dim=-1).clamp(min=1)
    return summed if CONFIG["score_normalization"] == "sum" else summed / lengths


def compute_calibration_scores(model, tokenizer, device) -> torch.Tensor:
    max_code_tokens = get_max_code_tokens(tokenizer)
    calibration_prompt = build_prompt(tokenizer, CALIBRATION_CODE, max_code_tokens)
    prompt_ids_batch = [tokenizer.encode(calibration_prompt, add_special_tokens=True)]

    class_scores = []
    for label_id in range(NUM_LABELS):
        phrase_scores = [
            score_candidate_batch(model, tokenizer, prompt_ids_batch, phrase, device)
            for phrase in get_verbalizer()[label_id]
        ]
        class_scores.append(torch.stack(phrase_scores, dim=0).max(dim=0).values.squeeze(0))

    scores = torch.stack(class_scores, dim=-1).detach()
    logging.info(f"Contextual calibration scores: {scores.cpu().tolist()}")
    return scores


def predict_with_calibrated_verbalizer(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    device,
    calibration_scores: torch.Tensor | None,
) -> torch.Tensor:
    prompt_ids_batch = []
    for ids, mask in zip(input_ids.cpu().tolist(), attention_mask.cpu().tolist()):
        prompt_ids_batch.append([token_id for token_id, keep in zip(ids, mask) if keep == 1])

    class_scores = []
    for label_id in range(NUM_LABELS):
        phrase_scores = [
            score_candidate_batch(model, tokenizer, prompt_ids_batch, phrase, device)
            for phrase in get_verbalizer()[label_id]
        ]
        class_scores.append(torch.stack(phrase_scores, dim=0).max(dim=0).values)

    scores = torch.stack(class_scores, dim=-1)
    if CONFIG["use_contextual_calibration"] and calibration_scores is not None:
        scores = scores - CONFIG["calibration_alpha"] * calibration_scores.to(device)
    return torch.argmax(scores, dim=-1)


def evaluate_model(
    model,
    data_loader,
    tokenizer,
    device,
    calibration_scores=None,
    best_val_score=-1,
    best_model_path="",
    stage="eval",
):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

    model.eval()
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for batch in data_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            preds = predict_with_calibrated_verbalizer(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                device,
                calibration_scores,
            )

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
        labels=list(range(NUM_LABELS)),
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

    if CONFIG["selection_metric"] == "weighted_macro_class3":
        selection_score = 0.70 * f1 + 0.20 * macro_f1 + 0.10 * class_f1[3]
    else:
        selection_score = f1
    logging.info(f"Selection score      : {selection_score:.4f}")

    if stage == "eval" and selection_score > best_val_score:
        best_val_score = selection_score
        torch.save(model.state_dict(), best_model_path)
        logging.info(f"New best model saved with selection score: {selection_score:.4f}")

    return acc, pre, rec, f1, best_val_score


def train_model(components, train_loader, val_loader, test_loader, device, save_model_path):
    model = components.model
    tokenizer = components.tokenizer
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

    best_val_score = -1
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
        calibration_scores = None
        if CONFIG["use_contextual_calibration"]:
            model.eval()
            with torch.inference_mode():
                calibration_scores = compute_calibration_scores(model, tokenizer, device)

        previous_best = best_val_score
        _, _, _, _, _, _, best_val_score = evaluate_model(
            model,
            val_loader,
            tokenizer,
            device,
            calibration_scores=calibration_scores,
            best_val_score=best_val_score,
            best_model_path=save_model_path,
            stage="eval",
        )
        if best_val_score > previous_best + CONFIG["min_delta"]:
            bad_epochs = 0
        else:
            bad_epochs += 1
            logging.info(
                f"Early stopping counter: {bad_epochs}/{CONFIG['early_stopping_patience']}"
            )
            if bad_epochs >= CONFIG["early_stopping_patience"]:
                logging.info("Early stopping triggered.")
                break

    del optimizer
    del scheduler
    torch.cuda.empty_cache()
    gc.collect()

    logging.info("\n--- testing ---")
    model.load_state_dict(torch.load(save_model_path, map_location="cpu"))
    calibration_scores = None
    if CONFIG["use_contextual_calibration"]:
        model.eval()
        with torch.inference_mode():
            calibration_scores = compute_calibration_scores(model, tokenizer, device)
    return evaluate_model(
        model,
        test_loader,
        tokenizer,
        device,
        calibration_scores=calibration_scores,
        stage="testing",
    )[:4]


def save_metrics(acc: float, pre: float, rec: float, f1: float, csv_path: str) -> None:
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Accuracy", "Precision", "Recall", "F1"])
        writer.writerow([round(acc, 4), round(pre, 4), round(rec, 4), round(f1, 4)])


def main() -> None:
    args = parse_args()
    apply_args(args)
    log_path, metrics_csv = setup_logging()

    logging.info(f"Run name: {CONFIG['run_name']}")
    logging.info(f"Model path: {CONFIG['model_path']}")
    logging.info(f"Data path: {CONFIG['data_path']}")
    logging.info(f"Log path: {log_path}")
    logging.info(f"Balanced sampling: {CONFIG['balanced_sampling']}")
    logging.info(f"Score normalization: {CONFIG['score_normalization']}")
    logging.info(f"Contextual calibration: {CONFIG['use_contextual_calibration']}")
    logging.info(f"Calibration alpha: {CONFIG['calibration_alpha']}")
    logging.info(f"Selection metric: {CONFIG['selection_metric']}")
    logging.info(f"Answer mode: {CONFIG['answer_mode']}")
    logging.info(f"Use few-shot examples: {CONFIG['use_few_shot']}")
    logging.info(f"Early stopping patience: {CONFIG['early_stopping_patience']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    if device.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    train_data, val_data, test_data = read_data(CONFIG["data_path"])
    total_acc, total_pre, total_rec, total_f1 = [], [], [], []

    for iter_num in range(CONFIG["iters"]):
        set_seed(CONFIG["seed"] + iter_num)
        save_model_path = os.path.join(CONFIG["output_dir"], f"{CONFIG['run_name']}_iter{iter_num + 1}_best.pth")

        components = load_model(CONFIG["model_path"])
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
        save_metrics(acc, pre, rec, f1, metrics_csv)

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

