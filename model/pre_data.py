import torch
import pandas as pd
from sklearn.utils import shuffle as reset
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, MixedTemplate, ManualVerbalizer, SoftTemplate, SoftVerbalizer

classes = [
    "0",
    "1",
    "2",
    "3",
]

def train_test_split(data, train_size=0.7, val_size=0.1, shuffle=True, random_state=None):
    if shuffle:
        data = reset(data, random_state=42)
    test_size = 1 - train_size - val_size
    train_data_size = int(len(data) * train_size)
    val_data_size = int((len(data) - train_data_size) * val_size)

    train = data[:train_data_size].reset_index(drop=True)
    val = data[train_data_size:(train_data_size + val_data_size)].reset_index(drop=True)
    test = data[(train_data_size + val_data_size):].reset_index(drop=True)

    return train, val, test

def load_pre_lm(model_name, model_path, model_type='', temp_types='hard', mapping='v2'):
    plm, tokenizer, _, WrapperClass = load_plm(model_name, model_path)


    if temp_types == 'hard':
        promptTemplate = ManualTemplate(
            text='The method has {"mask"} code smell. {"placeholder":"text_a"}',
            tokenizer=tokenizer,
        )
    elif temp_types == 'soft':
        promptTemplate = SoftTemplate(
            model=plm,
            tokenizer=tokenizer,
            num_tokens=10,
            text='{"soft"} {"soft"} {"soft"} {"mask"} {"soft"} {"soft"}. {"placeholder":"text_a"}'
        )
    else:
        promptTemplate = MixedTemplate(
            model=plm,
            tokenizer=tokenizer,
            text='{"soft"} {"soft"} {"soft"} {"mask"} code smell. {"placeholder":"text_a"}.'
        )

    
    if mapping == 'v1':
        promptVerbalizer = ManualVerbalizer(
            classes=classes,
            label_words={
                "0": ["no"],
                "1": ["Long Parameter List"],
                "2": ["Long Method"],
                "3": ["Long Method and Long Parameter List"],
            },
            tokenizer=tokenizer,
        )

    elif mapping == 'v2':
        promptVerbalizer = ManualVerbalizer(
            classes=classes,
            label_words={
                "0": ["no", "not", "zero"],
                "1": ["long parameter list", "lpl", "Long Parameter List", "Lpl"],
                "2": ["long method", "lm", "Long Method", "LM", "Lm"],
                "3": ["long method and long parameter list", "Long Method and Long Parameter List", "two", "all"],
            },
            tokenizer=tokenizer,
        )
    return plm, tokenizer, WrapperClass, promptTemplate, promptVerbalizer

def read_data(file_path, shuffle=True, generalization=False):
    df = pd.read_json(file_path)
    if generalization:
        dataset = [InputExample(guid=idx, text_a=code, label=label) for idx, (code, label) in
         enumerate(zip(df['newCode'], df['labels']))]
        if shuffle:
            dataset = reset(dataset, random_state=42)
        return dataset
    else:
        train, val, test = train_test_split(df)
        traindataset = [InputExample(guid=idx, text_a=code, label=label) for idx, (code, label) in enumerate(zip(train['newCode'], train['labels']))]
        valdataset = [InputExample(guid=idx, text_a=code, label=label) for idx, (code, label) in enumerate(zip(val['newCode'], val['labels']))]
        testdataset = [InputExample(guid=idx, text_a=code, label=label) for idx, (code, label) in enumerate(zip(test['newCode'], test['labels']))]
        if shuffle:
            traindataset = reset(traindataset, random_state=42)
            valdataset = reset(valdataset, random_state=42)
            testdataset = reset(testdataset, random_state=42)
        return traindataset, valdataset, testdataset

if __name__ == '__main__':
    file_path = '../data/LM-LPL.json'
    model_name = 'roberta'
    model_path = '../pre_model/UniXcoder'
    load_pre_lm(model_name, model_path)

