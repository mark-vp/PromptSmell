import argparse
import csv
import gc
import importlib.util
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEEPSEEK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def find_base_script() -> str:
    candidates = [
        os.environ.get("DEEPSEEK_BASE_SCRIPT", ""),
        os.path.join(SCRIPT_DIR, "deepseek_prompt_v3_light.py"),
        os.path.join(DEEPSEEK_DIR, "deepseek_prompt_v3_light.py"),
        os.path.join(os.getcwd(), "deepseek_prompt_v3_light.py"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    searched = "\n".join(f"  - {path}" for path in candidates if path)
    raise FileNotFoundError(
        "Base DeepSeek script not found. Please put deepseek_prompt_v3_light.py "
        "in the same directory as deepseek_few_shot_train.py, or set "
        f"DEEPSEEK_BASE_SCRIPT to its full path.\nSearched paths:\n{searched}"
    )


BASE_SCRIPT_PATH = find_base_script()


def find_project_root(start_dir: str) -> str:
    current = os.path.abspath(start_dir)
    while True:
        if os.path.exists(os.path.join(current, "model", "pre_data.py")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError("Cannot find project root containing model/pre_data.py")
        current = parent


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)


def load_base_module():
    spec = importlib.util.spec_from_file_location("deepseek_prompt_v3_light_base", BASE_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()


CONFIG = {
    "device_id": "0",
    "model_path": "pre_model/DeepSeek-Coder-1.3b-instruct",
    "data_dir": "data",
    "test_path": "data/LM-LPL.json",
    "output_dir": "results/deepseek_few_shot_train",
    "run_name": "DeepSeek_few_shot_train",
    "sample_files": ["64data.json", "256data.json", "512data.json", "1024data.json"],
    "include_zero_shot": True,
    "batch_size": 4,
    "epochs": 10,
    "iters": 5,
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
    "use_prompt_examples": False,
    "cleanup_checkpoints": True,
}


@dataclass
class CodeExample:
    guid: int
    text_a: str
    label: int


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Few-shot-size supervised training for DeepSeek prompt learning. "
            "This follows model/few_shot_train.py by using 64/256/512/1024 JSON files "
            "as training sets while keeping validation/test from LM-LPL.json."
        )
    )
    parser.add_argument("--model_path", default=CONFIG["model_path"])
    parser.add_argument("--data_dir", default=CONFIG["data_dir"])
    parser.add_argument("--test_path", default=CONFIG["test_path"])
    parser.add_argument("--output_dir", default=CONFIG["output_dir"])
    parser.add_argument("--run_name", default=CONFIG["run_name"])
    parser.add_argument("--device_id", default=CONFIG["device_id"])
    parser.add_argument("--sample_files", nargs="*", default=CONFIG["sample_files"])
    parser.add_argument("--skip_zero_shot", action="store_true")
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
    parser.add_argument("--use_prompt_examples", action="store_true")
    parser.add_argument(
        "--keep_checkpoints",
        action="store_true",
        help="Keep best-model .pth files after each test run. By default they are deleted to save disk space.",
    )
    parser.add_argument(
        "--selection_metric",
        choices=["weighted_f1", "weighted_macro_class3"],
        default=CONFIG["selection_metric"],
    )
    return parser.parse_args()


def apply_args(args) -> None:
    CONFIG["model_path"] = args.model_path
    CONFIG["data_dir"] = args.data_dir.rstrip("/")
    CONFIG["test_path"] = args.test_path
    CONFIG["output_dir"] = args.output_dir
    CONFIG["run_name"] = args.run_name
    CONFIG["device_id"] = args.device_id
    CONFIG["sample_files"] = args.sample_files
    CONFIG["include_zero_shot"] = not args.skip_zero_shot
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
    CONFIG["answer_mode"] = args.answer_mode
    CONFIG["early_stopping_patience"] = args.early_stopping_patience
    CONFIG["min_delta"] = args.min_delta
    CONFIG["use_prompt_examples"] = args.use_prompt_examples
    CONFIG["cleanup_checkpoints"] = not args.keep_checkpoints
    CONFIG["selection_metric"] = args.selection_metric

    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["device_id"]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    base.CONFIG["device_id"] = CONFIG["device_id"]
    base.CONFIG["model_path"] = CONFIG["model_path"]
    base.CONFIG["output_dir"] = CONFIG["output_dir"]
    base.CONFIG["batch_size"] = CONFIG["batch_size"]
    base.CONFIG["epochs"] = CONFIG["epochs"]
    base.CONFIG["iters"] = CONFIG["iters"]
    base.CONFIG["gradient_accumulation_steps"] = CONFIG["gradient_accumulation_steps"]
    base.CONFIG["seed"] = CONFIG["seed"]
    base.CONFIG["max_seq_length"] = CONFIG["max_seq_length"]
    base.CONFIG["lora_rank"] = CONFIG["lora_rank"]
    base.CONFIG["learning_rate"] = CONFIG["learning_rate"]
    base.CONFIG["weight_decay"] = CONFIG["weight_decay"]
    base.CONFIG["lora_dropout"] = CONFIG["lora_dropout"]
    base.CONFIG["balanced_sampling"] = CONFIG["balanced_sampling"]
    base.CONFIG["score_normalization"] = CONFIG["score_normalization"]
    base.CONFIG["use_contextual_calibration"] = CONFIG["use_contextual_calibration"]
    base.CONFIG["calibration_alpha"] = CONFIG["calibration_alpha"]
    base.CONFIG["selection_metric"] = CONFIG["selection_metric"]
    base.CONFIG["answer_mode"] = CONFIG["answer_mode"]
    base.CONFIG["early_stopping_patience"] = CONFIG["early_stopping_patience"]
    base.CONFIG["min_delta"] = CONFIG["min_delta"]
    base.CONFIG["use_few_shot"] = CONFIG["use_prompt_examples"]


def setup_logging() -> Tuple[str, str, str, str]:
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    log_path = os.path.join(CONFIG["output_dir"], f"{CONFIG['run_name']}.log")
    metrics_csv = os.path.join(CONFIG["output_dir"], f"{CONFIG['run_name']}_metrics.csv")
    summary_csv = os.path.join(CONFIG["output_dir"], f"{CONFIG['run_name']}_summary.csv")
    table_csv = os.path.join(CONFIG["output_dir"], f"{CONFIG['run_name']}_paper_table.csv")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path, metrics_csv, summary_csv, table_csv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_sample_file(path_or_name: str) -> str:
    if os.path.isabs(path_or_name):
        return path_or_name
    return os.path.join(CONFIG["data_dir"], path_or_name)


def load_small_train_data(file_path: str, seed: int = 42) -> List[CodeExample]:
    df = pd.read_json(file_path)
    required_columns = {"newCode", "labels"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{file_path} is missing required columns: {sorted(missing)}")
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return [
        CodeExample(guid=idx, text_a=str(code), label=int(label))
        for idx, (code, label) in enumerate(zip(df["newCode"], df["labels"]))
    ]


def label_distribution(data: Sequence[CodeExample]) -> Dict[int, int]:
    labels = [int(item.label) for item in data]
    return {label_id: labels.count(label_id) for label_id in range(base.NUM_LABELS)}


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    if device.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def save_iteration_metrics(
    csv_path: str,
    dataset_name: str,
    sample_size: int,
    acc: float,
    pre: float,
    rec: float,
    f1: float,
) -> None:
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "Dataset",
                    "SampleSize",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                ]
            )
        writer.writerow(
            [
                dataset_name,
                sample_size,
                round(acc, 4),
                round(pre, 4),
                round(rec, 4),
                round(f1, 4),
            ]
        )


def save_summary(summary_csv: str, dataset_name: str, sample_size: int, values: Dict[str, List[float]]) -> None:
    file_exists = os.path.isfile(summary_csv)
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "Dataset",
                    "SampleSize",
                    "Runs",
                    "AccuracyMean",
                    "AccuracyStd",
                    "PrecisionMean",
                    "PrecisionStd",
                    "RecallMean",
                    "RecallStd",
                    "F1Mean",
                    "F1Std",
                ]
            )
        writer.writerow(
            [
                dataset_name,
                sample_size,
                len(values["acc"]),
                round(float(np.mean(values["acc"])), 4),
                round(float(np.std(values["acc"])), 4),
                round(float(np.mean(values["pre"])), 4),
                round(float(np.std(values["pre"])), 4),
                round(float(np.mean(values["rec"])), 4),
                round(float(np.std(values["rec"])), 4),
                round(float(np.mean(values["f1"])), 4),
                round(float(np.std(values["f1"])), 4),
            ]
        )


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def save_paper_table_row(table_csv: str, sample_size: int, values: Dict[str, List[float]]) -> None:
    file_exists = os.path.isfile(table_csv)
    with open(table_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["# Training samples", "accuracy", "precision_w", "recall_w", "F1_w"])
        writer.writerow(
            [
                sample_size,
                format_percent(float(np.mean(values["acc"]))),
                format_percent(float(np.mean(values["pre"]))),
                format_percent(float(np.mean(values["rec"]))),
                format_percent(float(np.mean(values["f1"]))),
            ]
        )


def log_paper_table_row(sample_size: int, values: Dict[str, List[float]]) -> None:
    logging.info(
        "Paper table row | # Training samples: %s | accuracy: %s | precision_w: %s | recall_w: %s | F1_w: %s",
        sample_size,
        format_percent(float(np.mean(values["acc"]))),
        format_percent(float(np.mean(values["pre"]))),
        format_percent(float(np.mean(values["rec"]))),
        format_percent(float(np.mean(values["f1"]))),
    )


def cleanup_checkpoint(path: str) -> None:
    if not CONFIG["cleanup_checkpoints"]:
        return
    if os.path.exists(path):
        os.remove(path)


def log_final_results(dataset_name: str, values: Dict[str, List[float]]) -> None:
    logging.info(f"\n==================== final results: {dataset_name} ====================")
    logging.info(f"Accuracy           : {np.mean(values['acc']):.4f} +/- {np.std(values['acc']):.4f}")
    logging.info(f"Weighted Precision : {np.mean(values['pre']):.4f} +/- {np.std(values['pre']):.4f}")
    logging.info(f"Weighted Recall    : {np.mean(values['rec']):.4f} +/- {np.std(values['rec']):.4f}")
    logging.info(f"Weighted F1-score  : {np.mean(values['f1']):.4f} +/- {np.std(values['f1']):.4f}")


def evaluate_zero_shot(val_data, test_data, device, metrics_csv: str, summary_csv: str, table_csv: str) -> None:
    dataset_name = "zero_shot"
    logging.info(f"\n--------------------------- {dataset_name} ---------------------------")
    set_seed(CONFIG["seed"])
    components = base.load_model(CONFIG["model_path"])
    components.model.to(device)
    _, _, test_loader = base.create_dataloaders(val_data[:1], val_data, test_data, components.tokenizer)
    calibration_scores = None
    if CONFIG["use_contextual_calibration"]:
        components.model.eval()
        with torch.inference_mode():
            calibration_scores = base.compute_calibration_scores(components.model, components.tokenizer, device)
    acc, pre, rec, f1 = base.evaluate_model(
        components.model,
        test_loader,
        components.tokenizer,
        device,
        calibration_scores=calibration_scores,
        stage="testing",
    )[:4]
    values = {
        "acc": [acc],
        "pre": [pre],
        "rec": [rec],
        "f1": [f1],
    }
    save_iteration_metrics(metrics_csv, dataset_name, 0, acc, pre, rec, f1)
    save_summary(summary_csv, dataset_name, 0, values)
    save_paper_table_row(table_csv, 0, values)
    log_final_results(dataset_name, values)
    log_paper_table_row(0, values)
    del components, test_loader
    torch.cuda.empty_cache()
    gc.collect()


def run_few_shot_dataset(
    sample_path: str,
    val_data,
    test_data,
    device,
    metrics_csv: str,
    summary_csv: str,
    table_csv: str,
) -> None:
    dataset_name = os.path.splitext(os.path.basename(sample_path))[0]
    train_data = load_small_train_data(sample_path, seed=CONFIG["seed"])
    logging.info(f"\n--------------------------- {sample_path} ---------------------------")
    logging.info(f"Dataset name: {dataset_name}")
    logging.info(f"Train sample size: {len(train_data)}")
    logging.info(f"Train label distribution: {label_distribution(train_data)}")

    values = {"acc": [], "pre": [], "rec": [], "f1": []}

    for iter_num in range(CONFIG["iters"]):
        set_seed(CONFIG["seed"] + iter_num)
        save_model_path = os.path.join(
            CONFIG["output_dir"],
            f"{CONFIG['run_name']}_{dataset_name}_iter{iter_num + 1}_best.pth",
        )

        components = base.load_model(CONFIG["model_path"])
        train_loader, val_loader, test_loader = base.create_dataloaders(
            train_data,
            val_data,
            test_data,
            components.tokenizer,
        )
        acc, pre, rec, f1 = base.train_model(
            components,
            train_loader,
            val_loader,
            test_loader,
            device,
            save_model_path,
        )
        values["acc"].append(acc)
        values["pre"].append(pre)
        values["rec"].append(rec)
        values["f1"].append(f1)
        save_iteration_metrics(metrics_csv, dataset_name, len(train_data), acc, pre, rec, f1)
        cleanup_checkpoint(save_model_path)

        del components, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    save_summary(summary_csv, dataset_name, len(train_data), values)
    save_paper_table_row(table_csv, len(train_data), values)
    log_final_results(dataset_name, values)
    log_paper_table_row(len(train_data), values)


def main() -> None:
    args = parse_args()
    apply_args(args)
    log_path, metrics_csv, summary_csv, table_csv = setup_logging()

    logging.info(f"Run name: {CONFIG['run_name']}")
    logging.info(f"Base script: {BASE_SCRIPT_PATH}")
    logging.info(f"Model path: {CONFIG['model_path']}")
    logging.info(f"Data dir: {CONFIG['data_dir']}")
    logging.info(f"Validation/test source: {CONFIG['test_path']}")
    logging.info(f"Sample files: {CONFIG['sample_files']}")
    logging.info(f"Log path: {log_path}")
    logging.info(f"Metrics CSV: {metrics_csv}")
    logging.info(f"Summary CSV: {summary_csv}")
    logging.info(f"Paper table CSV: {table_csv}")
    logging.info(f"Few-shot design: supervised small training sets, not in-context examples")
    logging.info(f"Prompt examples enabled: {CONFIG['use_prompt_examples']}")
    logging.info(f"Cleanup checkpoints: {CONFIG['cleanup_checkpoints']}")
    logging.info(f"Answer mode: {CONFIG['answer_mode']}")
    logging.info(f"Selection metric: {CONFIG['selection_metric']}")
    logging.info(f"Epochs: {CONFIG['epochs']}")

    device = get_device()
    _, val_data, test_data = base.read_data(CONFIG["test_path"])
    logging.info(f"Validation size: {len(val_data)}")
    logging.info(f"Test size: {len(test_data)}")

    if CONFIG["include_zero_shot"]:
        evaluate_zero_shot(val_data, test_data, device, metrics_csv, summary_csv, table_csv)

    for sample_file in CONFIG["sample_files"]:
        sample_path = normalize_sample_file(sample_file)
        run_few_shot_dataset(sample_path, val_data, test_data, device, metrics_csv, summary_csv, table_csv)


if __name__ == "__main__":
    main()

