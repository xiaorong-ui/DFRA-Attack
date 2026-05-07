from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

@dataclass
class WandbConfig:
    entity: str = "???"  # fill your wandb entity
    project: str = "local_adversarial_attack"

@dataclass
class BlackboxConfig:
    model_name: str = "gpt4v"
    batch_size: int = 1
    timeout: int = 30

@dataclass
class DataConfig:
    batch_size: int = 1
    num_samples: int = 1000
    sample_mode: str = "interval"
    cle_data_path: str = "resources/images/bigscale"
    tgt_data_path: str = "resources/images/target_images"
    output: str = "./Ours"
    method: str = "DFRA"

@dataclass
class OptimConfig:
    alpha: float = 1.0
    epsilon: int = 8
    steps: int = 300

@dataclass
class ModelConfig:
    input_res: int = 336
    use_source_crop: bool = True
    use_target_crop: bool = True
    crop_scale: tuple = (0.5, 0.9)
    ensemble: bool = True
    device: str = "cuda:0"
    backbone: list = (
        "L336",
        "B16",
        "B32",
        "Laion",
    )

@dataclass
class DFRAAttackConfig:
    cluster_sequence: list = field(default_factory=lambda: [3, 5])
    mask_ratio_by_cluster: dict = field(default_factory=lambda: {3: 0.20, 5: 0.25})
    lambda_attn_by_cluster: dict = field(default_factory=lambda: {3: 0.1, 5: 0.1})
    lambda_rel: float = 0.5
    use_random_erasing: bool = True
    use_saliency_mask: bool = True
    mask_off_steps: int = 200

@dataclass
class MainConfig:
    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig()
    model: ModelConfig = ModelConfig()
    dfra_attack: DFRAAttackConfig = DFRAAttackConfig()
    wandb: WandbConfig = WandbConfig()
    blackbox: BlackboxConfig = BlackboxConfig()
    attack: str = "fgsm"

@dataclass
class Ensemble3ModelsConfig(MainConfig):
    data: DataConfig = DataConfig(batch_size=1)
    model: ModelConfig = ModelConfig(
        use_source_crop=True, use_target_crop=True, backbone=["B16", "B32", "Laion"]
    )

cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig)
cs.store(name="ensemble_3models", node=Ensemble3ModelsConfig)
