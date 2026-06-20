# Supplemental Evaluation Summary

- task_id: `effectiveness_task_20260618_153413`
- baseline_models: `deepseek-v4, deepseek-v4-pro, minimax`
- supplemental_models: `glm-4.7, kimi-k2, doubao`

## b1_eval

- questions: `11`
- models_after_merge: `5`
- top_bottom_gap: `0.095227`
- score_variance: `0.00102`

| Model | Overall | QA | MC |
|---|---:|---:|---:|
| deepseek-v4 | 0.466223 | 0.453556 | 0.5 |
| deepseek-v4-pro | 0.44218 | 0.420497 | 0.5 |
| doubao | 0.482407 | 0.47581 | 0.4 |
| glm-4.7 | 0.46641 | 0.453813 | 0.5 |
| minimax | 0.537407 | 0.363935 | 1.0 |

## bfull_eval

- questions: `24`
- models_after_merge: `5`
- top_bottom_gap: `0.067629`
- score_variance: `0.000496`

| Model | Overall | QA | MC |
|---|---:|---:|---:|
| deepseek-v4 | 0.477211 | 0.441836 | 0.545455 |
| deepseek-v4-pro | 0.440685 | 0.393136 | 0.545455 |
| doubao | 0.467968 | 0.421875 | 0.555556 |
| glm-4.7 | 0.459944 | 0.411252 | 0.5 |
| minimax | 0.508314 | 0.344419 | 1.0 |
