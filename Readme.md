# LoRA-MGPO

Code for **LoRA-MGPO: Mitigating Double Descent in LoRA Fine-Tuning via Momentum-Guided Perturbation Optimization**.

This work has been accepted to **Findings of EMNLP 2025**.

LoRA-MGPO reuses optimizer momentum states to construct LoRA weight-space perturbations and applies EMA-based adaptive perturbation normalization. The implementation is in `lora_mgpo/`.

## Installation

```bash
pip install -r requirements.txt
```

## Main Experiments

### NLU

T5-base on GLUE-style NLU tasks:

```bash
python run_exp.py ++dataset_name=mnli +peft=all model=t5base ++seed=42 +peft.lora_r=8 peft.lora_alpha=16 model.learning_rate=1e-4 model.max_length=128 model.real_batch_size=32 wandb.project=${your_project} wandb.name=lora_mgpo_nlu ++perturbation_method=mgpo ++mgpo_profile=nlu
```

### NLG

LLaMA-2-7B on generation tasks:

```bash
python run_exp.py ++dataset_name=meta_math +peft=all model=llama ++seed=42 +peft.lora_r=8 peft.lora_alpha=16 model.learning_rate=2e-5 model.max_length=1024 model.real_batch_size=32 wandb.project=${your_project} wandb.name=lora_mgpo_nlg ++perturbation_method=mgpo ++mgpo_profile=nlg
```

## Hyperparameters

| Profile | Model/Task | `rho` | `mu` | `beta` | LR | Max length | Batch size |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nlu` | T5-base / GLUE | 0.05 | 0.9 | 0.9 | 1e-4 | 128 | 32 |
| `nlg` | LLaMA-2-7B / NLG | 0.01 | 0.8 | 0.8 | 2e-5 | 1024 | 32 |

`mu` is used as AdamW `beta1` when `++perturbation_method=mgpo` is enabled. To override the paper profiles, pass explicit values such as `++rho=0.02 ++mgpo_mu=0.85 ++mgpo_beta=0.85`.

## Evaluation

```bash
python eval_gsm8k_batch.py results/${your_project}_meta_math/${wandb.name}/42/checkpoint-6250/
```

Additional scripts: `eval_gsm8k.py`, `eval_humaneval.py`, and `eval_mmlu.py`.

## Acknowledgement

This repository builds on code from [nblt/Flat-LoRA](https://github.com/nblt/Flat-LoRA) and [Outsider565/LoRA-GA](https://github.com/Outsider565/LoRA-GA). We thank the authors for releasing their implementations.

## Citation

```bibtex
@inproceedings{chang-etal-2025-lora,
    title = "{L}o{RA}-{MGPO}: Mitigating Double Descent in Low-Rank Adaptation via Momentum-Guided Perturbation Optimization",
    author = "Chang, Yupeng  and
      Guo, Chenlu  and
      Chang, Yi  and
      Wu, Yuan",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.34/",
    doi = "10.18653/v1/2025.findings-emnlp.34",
    pages = "648--659",
    ISBN = "979-8-89176-335-7"
}

```
