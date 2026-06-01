# LoRA-MGPO Module

This folder contains the LoRA-MGPO trainer and paper-aligned hyperparameter profiles.

| Profile | Scenario | rho | mu | beta |
| --- | --- | ---: | ---: | ---: |
| `nlu` | T5-base / GLUE | 0.05 | 0.9 | 0.9 |
| `nlg` | LLaMA-2-7B / NLG | 0.01 | 0.8 | 0.8 |

Use from the project root with `++perturbation_method=mgpo` and either `++mgpo_profile=nlu` or `++mgpo_profile=nlg`.
