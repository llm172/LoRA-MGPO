from copy import deepcopy
from typing import Any, Dict


LORA_MGPO_PROFILES: Dict[str, Dict[str, Any]] = {
    "nlu": {
        "rho": 0.05,
        "mu": 0.9,
        "beta": 0.9,
        "learning_rate": 1e-4,
        "max_length": 128,
        "macro_batch_size": 32,
        "description": "T5-base GLUE/NLU setting from the LoRA-MGPO paper.",
    },
    "nlg": {
        "rho": 0.01,
        "mu": 0.8,
        "beta": 0.8,
        "learning_rate": 2e-5,
        "max_length": 1024,
        "macro_batch_size": 32,
        "description": "LLaMA-2-7B NLG setting from the LoRA-MGPO paper.",
    },
}


def resolve_mgpo_hparams(profile: str, **overrides: Any) -> Dict[str, Any]:
    profile = (profile or "nlu").lower()
    if profile not in LORA_MGPO_PROFILES:
        raise ValueError(
            f"Unknown LoRA-MGPO profile '{profile}'. "
            f"Available profiles: {sorted(LORA_MGPO_PROFILES)}"
        )

    hparams = deepcopy(LORA_MGPO_PROFILES[profile])
    hparams["profile"] = profile
    for key, value in overrides.items():
        if value is not None:
            hparams[key] = value

    if hparams["rho"] <= 0:
        raise ValueError("LoRA-MGPO requires rho > 0.")
    if not 0 <= hparams["mu"] < 1:
        raise ValueError("LoRA-MGPO requires 0 <= mu < 1.")
    if not 0 <= hparams["beta"] < 1:
        raise ValueError("LoRA-MGPO requires 0 <= beta < 1.")
    return hparams
