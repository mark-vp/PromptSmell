import logging
import os
import random
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # Windows 环境下自动选择GPU
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

# 修复 openprompt 与新版本 transformers 的兼容性问题
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from transformers.generation_utils import GenerationMixin
except ImportError:
    from transformers.generation import utils
    sys.modules['transformers.generation_utils'] = utils

try:
    from transformers.tokenization_utils import PreTrainedTokenizer
except ImportError:
    from transformers.tokenization_utils_base import PreTrainedTokenizer
    import transformers.tokenization_utils_base as tokenization_utils
    sys.modules['transformers.tokenization_utils'] = tokenization_utils

import numpy as np
import torch
import torch.nn.functional as F
from openprompt import PromptForClassification, PromptDataLoader
from pre_data import read_data, load_pre_lm
from torch import optim
from tqdm import tqdm
from utils import eval_results, show_confusion_matrix, show_pr
from transformers import get_linear_schedule_with_warmup
from ptflops import get_model_complexity_info
import time

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def test_val_model(model, data_loader, device, best_val_f1, best_model_path='', types='val', is_save_cm=False, save_path=''):

    model.eval()
    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)
    all_probs_positive_class = []

    total_inference_time = 0
    num_samples = 0

    with torch.no_grad():
        for i, val_batch in enumerate(data_loader):
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            num_samples += val_batch['input_ids'].size(0)

            if device.type == 'cuda':
                torch.cuda.synchronize()  # 等待GPU完成之前的所有任务
            start_time = time.perf_counter()

            logits = model(val_batch)

            if device.type == 'cuda':
                torch.cuda.synchronize()  # 再次等待，确保推理任务已完成
            end_time = time.perf_counter()

            total_inference_time += (end_time - start_time)

            labels = val_batch['label']

            preds = torch.argmax(logits, dim=-1)
            probabilities = F.softmax(logits, dim=-1)

            if i != 0:
                j = len(all_preds)
                all_labels = np.insert(all_labels, j, labels.cpu().tolist(), axis=0)
                all_preds = np.insert(all_preds, j, preds.cpu().tolist(), axis=0)
                all_probs_positive_class = np.insert(all_probs_positive_class, j, probabilities[:, 1].cpu().tolist(),
                                                     axis=0)
            else:
                j = len(preds)
                all_labels = labels.cpu().tolist()
                all_preds = preds.cpu().tolist()
                all_probs_positive_class = probabilities[:, 1].cpu().tolist()

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs_positive = np.array(all_probs_positive_class)

    accuracy, weighted_precision, weighted_recall, weighted_f1, report, cm = eval_results(
        y_true, y_pred, y_probs_positive
    )

    logging.info("\n--- 评估指标 ---")
    logging.info(f"准确率 (Accuracy)              : {accuracy * 100:.2f}%")
    logging.info(f"加权查准率 (Weighted Precision): {weighted_precision:.4f}")
    logging.info(f"加权查全率 (Weighted Recall)   : {weighted_recall:.4f}")
    logging.info(f"加权F1分数 (Weighted F1-score) : {weighted_f1:.4f}")

    logging.info("--- 分类报告 (Classification Report) ---")
    logging.info(report)

    logging.info("\n--- 混淆矩阵 (Confusion Matrix) ---")
    if is_save_cm:
        show_confusion_matrix(cm, save_path)

    if types == 'val':
        if weighted_f1 > best_val_f1:
            best_val_f1 = weighted_f1
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved with F1: {best_val_f1:.4f}")

    avg_latency_ms = (total_inference_time / num_samples) * 1000
    throughput_sps = num_samples / total_inference_time

    return accuracy, weighted_precision, weighted_recall, weighted_f1, avg_latency_ms, throughput_sps

def train_model(epochs, traindata_loader, valdata_loader, testdata_loader, model, device, best_model_path):
    model.to(device)
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm', 'LayerNorm.weight']
    optimizer_growped_paramters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Trainable parameters: {trainable_params}")
    logging.info(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=optimizer_growped_paramters, lr=1e-5, weight_decay=0.01)
    num_training_steps = len(traindata_loader) * epochs
    warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    best_val_f1 = -1

    for epoch in range(epochs):
        total_loss = 0.
        for i, train_batch in enumerate(tqdm(traindata_loader)):
            train_batch = {k: v.to(device) for k, v in train_batch.items()}

            logits = model(train_batch)
            loss = loss_func(logits, train_batch['label'])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            total_loss += loss.item()

        logging.info(f"Epoch {epoch + 1}, Loss: {total_loss / len(traindata_loader)}")

        logging.info("\n--- eval ---")
        (val_accuracy, val_weighted_precision, val_weighted_recall, val_weighted_f1,
         avg_latency_ms, throughput_sps) = test_val_model(
            model, data_loader=valdata_loader, device=device, best_val_f1=best_val_f1,
            best_model_path=best_model_path, types='val')


    logging.info("\n--- testing ---")
    model.load_state_dict(torch.load(best_model_path))
    (test_accuracy, test_weighted_precision, test_weighted_recall, test_weighted_f1,
     avg_latency_ms, throughput_sps) = test_val_model(
        model, testdata_loader, device, best_val_f1=best_val_f1, types='test')

    return test_accuracy, test_weighted_precision, test_weighted_recall, test_weighted_f1, avg_latency_ms, throughput_sps


def create_model(train, val, test, model_name, model_path):
    set_seed(42)

    plm, tokenizer, WrapperClass, promptTemplate, promptVerbalizer = load_pre_lm(model_name, model_path)

    traindata_loader = PromptDataLoader(
        dataset=train,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=8,
        shuffle=True
    )
    valdata_loader = PromptDataLoader(
        dataset=val,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=8,
        shuffle=True
    )

    testdata_loader = PromptDataLoader(
        dataset=test,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=8,
        shuffle=True
    )

    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=promptVerbalizer,
    )

    return traindata_loader, valdata_loader, testdata_loader, promptModel

if __name__ == "__main__":
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    file_path = os.path.join(BASE_DIR, 'data', 'LM-LPL.json')
    model_name = 'roberta'
    model_path = os.path.join(BASE_DIR, 'pre_model', 'UniXcoder')
    iters = 1  # 简化测试，只运行1次
    epochs = 2  # 简化测试，只训练2个epoch
    total_acc = []
    total_pre = []
    total_rec = []
    total_f1 = []
    total_latency, total_throughput = [], []

    log_path = os.path.join(BASE_DIR, 'results', 'p1_v2.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("GPU not available, using CPU.")

    train, val, test = read_data(file_path)

    best_model_path = os.path.join(BASE_DIR, 'save_dict', 'p1_v2_save_model.pth')
    traindata_loader, valdata_loader, testdata_loader, promptModel = create_model(train, val, test, model_name, model_path)

    for iter in range(iters):
        logging.info(f'-----------------  {iter}  -----------------')

        (test_accuracy, test_weighted_precision, test_weighted_recall, test_weighted_f1,
         avg_latency_ms, throughput_sps) = train_model(
            epochs, traindata_loader, valdata_loader, testdata_loader, promptModel, device=device,
            best_model_path=best_model_path
        )

        total_acc.append(test_accuracy)
        total_pre.append(test_weighted_precision)
        total_rec.append(test_weighted_recall)
        total_f1.append(test_weighted_f1)

    mean_acc = np.mean(total_acc)
    mean_pre = np.mean(total_pre)
    mean_rec = np.mean(total_rec)
    mean_f1 = np.mean(total_f1)

    std_acc = np.std(total_acc)
    std_pre = np.std(total_pre)
    std_rec = np.std(total_rec)
    std_f1 = np.std(total_f1)
    mean_latency, std_latency = np.mean(total_latency), np.std(total_latency)
    mean_throughput, std_throughput = np.mean(total_throughput), np.std(total_throughput)

    logging.info(f"\n\n{'=' * 20} 实验最终结果 ({iters}次运行) {'=' * 20}\n")
    # print(f"模型理论复杂度:")
    # print(f"  - 参数量 (Params)      : {params}")
    # print(f"  - 计算量 (FLOPs)       : {gflops:.2f} GFLOPs (针对单一样本)\n")
    logging.info("模型实际性能 (Mean ± Std Dev):")
    logging.info("----------------------------------------------------------------------")
    logging.info(f"  - Accuracy             : {mean_acc:.4f} ± {std_acc:.4f}")
    logging.info(f"  - Weighted Precision   : {mean_pre:.4f} ± {std_pre:.4f}")
    logging.info(f"  - Weighted Recall      : {mean_rec:.4f} ± {std_rec:.4f}")
    logging.info(f"  - Weighted F1-score    : {mean_f1:.4f} ± {std_f1:.4f}")
    logging.info("----------------------------------------------------------------------")
    logging.info(f"  - 推理延迟 (Latency)     : {mean_latency:.2f} ± {std_latency:.2f} ms/sample")
    logging.info(f"  - 推理吞吐量 (Throughput) : {mean_throughput:.2f} ± {std_throughput:.2f} samples/sec")
    logging.info("\n" + "=" * 60)
