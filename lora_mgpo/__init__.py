from .config import LORA_MGPO_PROFILES, resolve_mgpo_hparams

__all__ = ["LORA_MGPO_PROFILES", "LoraMGPOTrainer", "resolve_mgpo_hparams"]


def __getattr__(name):
    if name == "LoraMGPOTrainer":
        from .trainer import LoraMGPOTrainer

        return LoraMGPOTrainer
    raise AttributeError(name)
