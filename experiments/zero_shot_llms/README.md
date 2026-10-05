# Zero-shot SOTA LLM comparison

This directory contains the direct zero-shot comparison reported in RQ5. All four models received the same test-set source view and task prompt. No labeled examples or gold labels were included in the client requests. Predictions use the same four-class mapping and evaluation metrics as PromptSmell.

| Model | Accuracy | Weighted precision | Weighted recall | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| GPT-5.6 Sol | 63.17% | 75.11% | 63.17% | 59.97% |
| GPT-6 Astra | 71.75% | 80.46% | 71.75% | 66.74% |
| DeepSeek V4 Pro | 75.09% | 81.31% | 75.09% | 72.83% |
| Gemini 3.8 Flash | 74.83% | 80.98% | 74.83% | 71.91% |

These are one-round results. GPT and Gemini were accessed through Responses-compatible gateways; DeepSeek was accessed through Volcengine Ark. The response-reported model IDs are recorded in `protocol.json`. The client prompt and source view were held fixed, while provider-side instructions and serving routes may differ. The table describes the observed results under those conditions.

## Files

- `data/test_matched_decoder.jsonl`: source code supplied to the models, with the same decoder-visible code limit used in PromptSmell's RQ5 evaluation.
- `data/gold_test.jsonl`: labels kept separate from inference.
- `prompts/system.txt` and `prompts/user.txt`: exact client-supplied task prompts.
- `predictions/*.csv`: complete first-round predictions. The `gold` and `correct` columns were added after inference.
- `protocol.json`: model settings, class mapping, and hashes of the frozen inputs and predictions.
- `usage.csv`: provider-reported input, cached-input, and output token totals. It does not report account charges.
- `scripts/evaluate_predictions.py`: four-class scoring from the released predictions and labels.
- `scripts/run_zero_shot.py`: an optional caller using the released client-side prompt and response parsing.

## Check the results

From the repository root, run:

```bash
python experiments/zero_shot_llms/scripts/evaluate_predictions.py
```

The script checks file hashes and sample IDs, then recomputes accuracy, weighted precision, weighted recall, and weighted F1. Invalid outputs remain in the test denominator.

The caller runs offline unless `--execute` is supplied. A live rerun needs credentials and an endpoint authorized for the chosen model. The original gateway's full service-side context is not part of this client package, so a rerun through another route may differ.
