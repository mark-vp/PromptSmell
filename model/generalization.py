import os
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
import torch
from pre_data import read_data, load_pre_lm
from main_train import create_model, test_val_model
from openprompt import PromptDataLoader, PromptForClassification
import logging
import sys

# generalization_results
log_path = '/home/lhy/storage/code/promptsmell/results/# generalization_results.log'
data_path = '/home/lhy/storage/code/promptsmell/data/test'
model_name = 'roberta'
model_path = '/home/lhy/storage/code/promptsmell/pre_model/UniXcoder'
best_model_path = '/home/lhy/storage/code/promptsmell/save_dict/p1_v2_save_model.pth'
save_path = '/home/lhy/storage/code/promptsmell/cm_results'

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
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU.")


data_list = os.listdir(data_path)

for path in data_list:
    name = path.split('.')[0] + '.png'
    print(f'------------ {path} ------------')
    file_path = os.path.join(data_path, path)
    png_path = os.path.join(save_path, name)
    test = read_data(file_path, generalization=True)

    plm, tokenizer, WrapperClass, promptTemplate, promptVerbalizer = load_pre_lm(model_name, 
                                                                                 model_path, )

    data_loader = PromptDataLoader(
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

    promptModel.load_state_dict(torch.load(best_model_path))
    promptModel.to(device)
    test_accuracy, test_weighted_precision, test_weighted_recall, test_weighted_f1, avg_latency_ms, throughput_sps = test_val_model(
        promptModel, data_loader, device, best_val_f1=-1, types='test', is_save_cm=True, save_path=png_path)
