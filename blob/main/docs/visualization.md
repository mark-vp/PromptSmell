Experimental Visualizations

This document provides supplementary visualizations for the PromptSmell experiments. These visualizations are intended to support the experimental results reported in the paper, especially the cross-project evaluation, fine-tuning comparison, baseline comparison, backbone comparison, few-shot analysis, and prompt-template analysis.

1. Overview

The paper evaluates PromptSmell using the following research questions:

RQ	Description
RQ1	How effective is PromptSmell on real-world projects?
RQ2	How effective is PromptSmell compared with fine-tuning?
RQ3	How effective is PromptSmell compared with representative baselines?
RQ4	How effective is PromptSmell under encoder-based and decoder-based settings?
RQ5	How effective is PromptSmell in zero-shot and few-shot settings?
RQ6	What is the impact of prompt templates and answer mappings on PromptSmell?

The evaluation metrics include accuracy, weighted precision, weighted recall, and weighted F1.

2. RQ1: Cross-project Results on Real-world Projects

RQ1 evaluates PromptSmell on six unseen real-world projects:

Drjava
Filecrush
Freeplane
JGroups
Nutch
PMD

The results compare two architecture settings:

Encoder-style PLM: PromptSmell with UniXcoder.
Decoder-based code LLM: PromptSmell with DeepSeek-Coder-1.3B-Instruct.
2.1 Cross-project Performance Table
Architecture	Project	Accuracy	Precisionw	Recallw	F1w
UniXcoder	Drjava	91.23%	98.56%	91.23%	94.39%
UniXcoder	Filecrush	93.46%	98.35%	93.46%	95.84%
UniXcoder	Freeplane	93.62%	97.78%	93.62%	95.22%
UniXcoder	JGroups	90.89%	97.45%	90.89%	93.55%
UniXcoder	Nutch	86.01%	97.42%	86.01%	90.84%
UniXcoder	PMD	93.87%	99.25%	93.87%	96.27%
UniXcoder	Average	91.51%	98.13%	91.51%	94.35%
DeepSeek-Coder	Drjava	92.99%	98.68%	92.99%	95.50%
DeepSeek-Coder	Filecrush	96.73%	98.13%	96.73%	97.31%
DeepSeek-Coder	Freeplane	96.63%	98.46%	96.63%	97.27%
DeepSeek-Coder	JGroups	95.40%	98.15%	95.40%	96.37%
DeepSeek-Coder	Nutch	88.72%	98.32%	88.72%	92.94%
DeepSeek-Coder	PMD	95.74%	99.50%	95.74%	97.50%
DeepSeek-Coder	Average	94.37%	98.54%	94.37%	96.15%

