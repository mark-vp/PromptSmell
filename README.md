# PromptSmell

PromptSmell is a prompt-learning framework for detecting Long Method and Long Parameter List code smells in Java methods. This repository contains the public code, dataset files, and experiment logs used for reproducibility.

## Project Structure

```text
data/
  LM-LPL.json              Main dataset (6,985 method-level samples)
  64data.json              Few-shot training subset
  256data.json             Few-shot training subset
  512data.json             Few-shot training subset
  1024data.json            Few-shot training subset
  test/                    Cross-project test sets
model/
  main_train.py            Main prompt-learning script with UniXcoder
  llm_prompt.py            Prompt learning with DeepSeek-Coder
  llm_prompt_1024.py       DeepSeek-Coder prompt learning with 1024-token input
  llm_finetune.py          Fine-tuning baseline with DeepSeek-Coder
  few_shot_train.py        Few-shot experiments with UniXcoder
  few_shot_train_d.py      Few-shot experiments with DeepSeek-Coder
  others_train.py          Baseline scripts
  generalization.py        Cross-project evaluation
  pre_data.py              Data loading and prompt/verbalizer utilities
  utils.py                 Evaluation and visualization utilities
  cm.py                    Confusion matrix visualization
results/                   Released experiment logs
cm_results/                Confusion matrix plots
blog/main/docs/            Supplementary documentation
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU for the large-model experiments
- `openprompt`, `transformers`, `peft`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, and `seaborn`

Install the required packages according to your local Python/CUDA environment. A minimal dependency file is provided as `requirements.txt`. Pre-trained models are not included in this repository.

## Dataset

The dataset contains Java methods labeled with two code smell types:

| Label | Long Parameter List | Long Method | Description |
| --- | --- | --- | --- |
| 0 | False | False | No target smell |
| 1 | True | False | Long Parameter List only |
| 2 | False | True | Long Method only |
| 3 | True | True | Both smells |

See `blog/main/docs/data_preparation.md` for dataset statistics.

## Pre-trained Models

```text
pre_model/
  UniXcoder/
  DeepSeek-Coder-1.3b-instruct/
  BERT/
  T5/
  GraphCodeBERT/
  CodeLlama-7b-instruct/
```

## Running Experiments

The main configurable arguments include `--model_path`, `--data_path`, `--data_dir`, `--epochs`, `--batch_size`, `--learning_rate`, and `--max_seq_length`, depending on the script.

## Results

Released logs and plots are stored in `results/` and `cm_results/`. The released logs focus on prediction metrics such as accuracy, weighted precision, weighted recall, and weighted F1.

## License

This project is released for research purposes.

## Citation

If you use this repository, please cite the corresponding PromptSmell paper.
