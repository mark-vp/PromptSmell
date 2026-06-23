# Data Preparation and Label Distribution

This file summarizes the dataset format, label-combination strategy, and label distribution used by PromptSmell. It is intended as supplementary repository information for the dataset description in the paper.

## Dataset Files

The main dataset and auxiliary evaluation files are stored in:

```text
data/
```

The most relevant files are:

| File or Directory | Description |
| --- | --- |
| `LM-LPL.json` | Main dataset used for training, validation, and testing. |
| `64data.json`, `256data.json`, `512data.json`, `1024data.json` | Small training subsets used in few-shot experiments. |
| `data/test/` | Cross-project test sets from unseen projects. |

Each JSON file stores method-level Java code and its corresponding label. The model receives the source-code text as input, while the label is used as the supervised target.

## Label Combination Strategy

The adopted dataset focuses on two representative code smell types:

| Code Smell | Abbreviation |
| --- | --- |
| Long Parameter List | LPL |
| Long Method | LM |

The original multi-label code smell attributes are converted into a single multi-class label through label combination. This allows the classifier to model co-occurrence relationships between smells in a unified label space.

For the two-smell setting used in this repository, the mapping is:

| Combined Label | Long Parameter List | Long Method | Meaning |
| --- | --- | --- | --- |
| `0` | False | False | No target smell. |
| `1` | True | False | Long Parameter List only. |
| `2` | False | True | Long Method only. |
| `3` | True | True | Both Long Parameter List and Long Method. |

This mapping is consistent with the label space used by the prompt verbalizer and the downstream classifier.


## Cross-Project Test Sets

The unseen project datasets are stored in `data/test/`:

| Project File | Samples |
| --- | ---: |
| `Drjava1.json` | 19,825 |
| `Filecrush1.json` | 367 |
| `Freeplane1.json` | 15,718 |
| `JGroups1.json` | 12,432 |
| `Nutch1.json` | 3,696 |
| `PMD1.json` | 14,427 |


## Extension to Additional Smells

PromptSmell can be extended to additional code smell types by following the same preparation procedure:

1. Define the target smell categories and the corresponding detection or labeling rules.
2. Convert the resulting multi-label attributes into a combined label space.
3. Add suitable label descriptors or verbalizer words for the newly introduced smell categories.
4. Use the prepared method-level code and labels in the same training and evaluation pipeline.

This extension keeps the framework consistent while allowing the label space to represent new smell combinations.

