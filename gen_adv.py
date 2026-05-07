# gen_adv.py - Generate adversarial images only (Phase 1)
# Generates adversarial images and saves a manifest.json for later evaluation

import os
import json
import random
import numpy as np
import torch
import torchvision
from PIL import Image
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Optional, Tuple, Dict, Any
from tqdm import tqdm
from torch import nn
from datetime import datetime
import shutil
from pathlib import Path

from config_schema import MainConfig
from surrogates import (
    ClipB16FeatureExtractor,
    ClipL336FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
    EnsembleFeatureExtractor_ot,
    EnsembleFeatureLoss_OT_dfra_attack,
)
from utils import (
    ensure_dir,
    get_output_paths,
)

# -----------------------
# BACKBONE MAP
# -----------------------
BACKBONE_MAP = {
    "L336": ClipL336FeatureExtractor,
    "B16": ClipB16FeatureExtractor,
    "B32": ClipB32FeatureExtractor,
    "Laion": ClipLaionFeatureExtractor
}

VALID_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".JPEG"]


def _get_cluster_value(mapping: Dict[Any, Any], cluster_num: int, name: str):
    if cluster_num in mapping:
        return mapping[cluster_num]
    cluster_key = str(cluster_num)
    if cluster_key in mapping:
        return mapping[cluster_key]
    raise KeyError(f"Missing {name} for cluster={cluster_num}. Available keys: {list(mapping.keys())}")


def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


import torchvision.transforms as transforms
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)


def log_metrics(pbar, metrics, img_index, epoch=None):
    pbar_metrics = {
        k: f"{v:.5f}" if "sim" in k else f"{v:.3f}" for k, v in metrics.items()
    }
    pbar.set_postfix(pbar_metrics)


def input_diversity(x: torch.Tensor, input_res: int = 224, prob: float = 0.5) -> torch.Tensor:
    if random.random() > prob:
        return x
    low = int(input_res * 0.8)
    rnd = random.randint(low, input_res)
    x_resized = torch.nn.functional.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False)
    pad_h = input_res - rnd
    pad_w = input_res - rnd
    pad_top = random.randint(0, pad_h)
    pad_left = random.randint(0, pad_w)
    return torch.nn.functional.pad(x_resized, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), value=0)


def build_backbone_models(cfg: MainConfig):
    if not cfg.model.ensemble and len(cfg.model.backbone) > 1:
        raise ValueError("When ensemble=False, only one backbone can be specified")

    models = []
    for backbone_name in cfg.model.backbone:
        if backbone_name not in BACKBONE_MAP:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        model_class = BACKBONE_MAP[backbone_name]
        model = model_class().eval().to(cfg.model.device).requires_grad_(False)
        models.append(model)
    return models


def get_models_ot_with_cluster(
    cfg: MainConfig,
    cluster_number: int,
    use_random_erasing: bool = False,
    erasing_scale=(0.12, 0.12),
    lambda_attn: float = 0.1,
    lambda_rel: float = 0.5,
    use_saliency_mask: bool = True,
    use_mask: bool = True,
    models=None,
):
    if not cfg.model.ensemble and len(cfg.model.backbone) > 1:
        raise ValueError("When ensemble=False, only one backbone can be specified")
    if models is None:
        models = build_backbone_models(cfg)

    if cfg.model.ensemble:
        ensemble_extractor = EnsembleFeatureExtractor_ot(models, cluster_number=cluster_number)
    else:
        ensemble_extractor = models[0]

    ensemble_loss = EnsembleFeatureLoss_OT_dfra_attack(
        models,
        cluster_number=cluster_number,
        use_random_erasing=use_random_erasing,
        erasing_prob=1.0,
        use_saliency_mask=use_saliency_mask,
        erasing_scale=erasing_scale,
        lambda_attn=lambda_attn,
        lambda_rel=lambda_rel,
        use_mask=use_mask,
    )
    return ensemble_extractor, models, ensemble_loss


def fgsm_attack(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
):
    if hasattr(ensemble_loss, "reset_attack_state"):
        ensemble_loss.reset_attack_state()

    delta = torch.zeros_like(image_org, requires_grad=True)
    pbar = tqdm(range(cfg.optim.steps), desc=f"Attack progress")

    original_use_erasing = getattr(ensemble_loss, 'use_random_erasing', False)
    mask_off_start = max(0, cfg.optim.steps - int(getattr(cfg.dfra_attack, "mask_off_steps", 200)))

    fixed_target_view = target_crop(image_tgt) if target_crop is not None else image_tgt

    ensemble_loss.use_random_erasing = original_use_erasing
    with torch.no_grad():
        ensemble_loss.set_ground_truth(fixed_target_view)
    fixed_mask_indices = list(ensemble_loss.current_mask_indices)
    fixed_mask_grid = ensemble_loss.current_mask_grid_shape
    fixed_gt_masked = list(ensemble_loss.ground_truth)
    fixed_gt_local_masked = list(ensemble_loss.ground_truth_local)
    fixed_gt_attn_masked = list(ensemble_loss.ground_truth_attention)

    ensemble_loss.use_random_erasing = False
    with torch.no_grad():
        ensemble_loss.set_ground_truth(fixed_target_view)
    fixed_gt_full = list(ensemble_loss.ground_truth)
    fixed_gt_local_full = list(ensemble_loss.ground_truth_local)
    fixed_gt_attn_full = list(ensemble_loss.ground_truth_attention)

    for epoch in pbar:
        use_mask = original_use_erasing and (epoch < mask_off_start)
        ensemble_loss.use_random_erasing = use_mask
        if use_mask:
            ensemble_loss.ground_truth[:] = fixed_gt_masked
            ensemble_loss.ground_truth_local[:] = fixed_gt_local_masked
            ensemble_loss.ground_truth_attention[:] = fixed_gt_attn_masked
            ensemble_loss.current_mask_indices = list(fixed_mask_indices)
        else:
            ensemble_loss.ground_truth[:] = fixed_gt_full
            ensemble_loss.ground_truth_local[:] = fixed_gt_local_full
            ensemble_loss.ground_truth_attention[:] = fixed_gt_attn_full
            ensemble_loss.current_mask_indices = []
        ensemble_loss.current_mask_grid_shape = fixed_mask_grid

        adv_image = image_org + delta
        adv_image_masked = ensemble_loss.apply_current_mask_to_image(adv_image)
        adv_features, adv_features_local, adv_features_attention = ensemble_extractor(adv_image_masked)

        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        if cfg.model.use_source_crop and source_crop is not None:
            local_cropped = source_crop(adv_image)
            local_cropped_masked = ensemble_loss.apply_current_mask_to_image(local_cropped)
            local_cropped_diverse = input_diversity(local_cropped_masked, cfg.model.input_res)
            local_features, local_features_local, local_features_attention = ensemble_extractor(local_cropped_diverse)
            local_sim = ensemble_loss(local_features, local_features_local, local_features_attention)
            loss = local_sim
            metrics["global_similarity"] = local_sim.item()
        else:
            adv_image_diverse = input_diversity(adv_image_masked, cfg.model.input_res)
            global_sim = ensemble_loss(*ensemble_extractor(adv_image_diverse))
            loss = global_sim
            metrics["global_similarity"] = global_sim.item()

        log_metrics(pbar, metrics, img_index, epoch)
        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        delta.data = torch.clamp(
            delta + cfg.optim.alpha * torch.sign(grad),
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )

    adv_image = image_org + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    final_metrics = {
        "max_delta": torch.max(torch.abs(delta)).item(),
        "mean_delta": torch.mean(torch.abs(delta)).item(),
    }
    log_metrics(pbar, final_metrics, img_index)

    ensemble_loss.use_random_erasing = original_use_erasing

    return adv_image


def mifgsm_attack(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
):
    if hasattr(ensemble_loss, "reset_attack_state"):
        ensemble_loss.reset_attack_state()
    delta = torch.zeros_like(image_org, requires_grad=True)
    momentum = torch.zeros_like(image_org, requires_grad=False)
    pbar = tqdm(range(cfg.optim.steps), desc=f"Attack progress")

    original_use_erasing = getattr(ensemble_loss, 'use_random_erasing', False)
    mask_off_start = max(0, cfg.optim.steps - int(getattr(cfg.dfra_attack, "mask_off_steps", 100)))

    with torch.no_grad():
        fixed_target_view = target_crop(image_tgt) if target_crop is not None else image_tgt
        ensemble_loss.set_ground_truth(fixed_target_view)
    fixed_mask_indices = list(ensemble_loss.current_mask_indices)
    fixed_mask_grid = ensemble_loss.current_mask_grid_shape
    fixed_ground_truth = list(ensemble_loss.ground_truth)
    fixed_ground_truth_local = list(ensemble_loss.ground_truth_local)
    fixed_ground_truth_attention = list(ensemble_loss.ground_truth_attention)

    for epoch in pbar:
        use_mask = original_use_erasing and (epoch < mask_off_start)
        ensemble_loss.use_random_erasing = use_mask
        ensemble_loss.ground_truth[:] = fixed_ground_truth
        ensemble_loss.ground_truth_local[:] = fixed_ground_truth_local
        ensemble_loss.ground_truth_attention[:] = fixed_ground_truth_attention
        ensemble_loss.current_mask_indices = list(fixed_mask_indices) if use_mask else []
        ensemble_loss.current_mask_grid_shape = fixed_mask_grid

        adv_image = image_org + delta
        adv_image_masked = ensemble_loss.apply_current_mask_to_image(adv_image)
        adv_features, adv_features_local, adv_features_attention = ensemble_extractor(adv_image_masked)

        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        if cfg.model.use_source_crop and source_crop is not None:
            local_cropped = source_crop(adv_image)
            local_cropped_masked = ensemble_loss.apply_current_mask_to_image(local_cropped)
            local_features, local_features_local, local_features_attention = ensemble_extractor(local_cropped_masked)
            local_sim = ensemble_loss(local_features, local_features_local, local_features_attention)
            loss = local_sim
            metrics["global_similarity"] = local_sim.item()
        else:
            global_sim = ensemble_loss(adv_features, adv_features_local, adv_features_attention)
            loss = global_sim
            metrics["global_similarity"] = global_sim.item()

        log_metrics(pbar, metrics, img_index, epoch)
        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        momentum = momentum * 0.9 + grad
        delta.data = torch.clamp(
            delta + cfg.optim.alpha * torch.sign(momentum),
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )

    adv_image = image_org + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    final_metrics = {
        "max_delta": torch.max(torch.abs(delta)).item(),
        "mean_delta": torch.mean(torch.abs(delta)).item(),
    }
    log_metrics(pbar, final_metrics, img_index)

    ensemble_loss.use_random_erasing = original_use_erasing

    return adv_image


def pgd_attack(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
):
    if hasattr(ensemble_loss, "reset_attack_state"):
        ensemble_loss.reset_attack_state()
    delta = torch.zeros_like(image_org, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=cfg.optim.alpha)
    pbar = tqdm(range(cfg.optim.steps), desc=f"Attack progress")

    original_use_erasing = getattr(ensemble_loss, 'use_random_erasing', False)
    mask_off_start = max(0, cfg.optim.steps - int(getattr(cfg.dfra_attack, "mask_off_steps", 100)))

    with torch.no_grad():
        fixed_target_view = target_crop(image_tgt) if target_crop is not None else image_tgt
        ensemble_loss.set_ground_truth(fixed_target_view)
    fixed_mask_indices = list(ensemble_loss.current_mask_indices)
    fixed_mask_grid = ensemble_loss.current_mask_grid_shape
    fixed_ground_truth = list(ensemble_loss.ground_truth)
    fixed_ground_truth_local = list(ensemble_loss.ground_truth_local)
    fixed_ground_truth_attention = list(ensemble_loss.ground_truth_attention)

    for epoch in pbar:
        use_mask = original_use_erasing and (epoch < mask_off_start)
        ensemble_loss.use_random_erasing = use_mask
        ensemble_loss.ground_truth[:] = fixed_ground_truth
        ensemble_loss.ground_truth_local[:] = fixed_ground_truth_local
        ensemble_loss.ground_truth_attention[:] = fixed_ground_truth_attention
        ensemble_loss.current_mask_indices = list(fixed_mask_indices) if use_mask else []
        ensemble_loss.current_mask_grid_shape = fixed_mask_grid

        adv_image = image_org + delta
        adv_image_masked = ensemble_loss.apply_current_mask_to_image(adv_image)
        adv_features, adv_features_local, adv_features_attention = ensemble_extractor(adv_image_masked)

        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        if cfg.model.use_source_crop and source_crop is not None:
            local_cropped = source_crop(adv_image)
            local_cropped_masked = ensemble_loss.apply_current_mask_to_image(local_cropped)
            local_features, local_features_local, local_features_attention = ensemble_extractor(local_cropped_masked)
            local_sim = ensemble_loss(local_features, local_features_local, local_features_attention)
            loss = -local_sim
            metrics["global_similarity"] = local_sim.item()
        else:
            global_sim = ensemble_loss(adv_features, adv_features_local, adv_features_attention)
            loss = -global_sim
            metrics["global_similarity"] = global_sim.item()

        log_metrics(pbar, metrics, img_index, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        delta.data = torch.clamp(
            delta,
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )

    adv_image = image_org + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    final_metrics = {
        "max_delta": torch.max(torch.abs(delta)).item(),
        "mean_delta": torch.mean(torch.abs(delta)).item(),
    }
    log_metrics(pbar, final_metrics, img_index)

    ensemble_loss.use_random_erasing = original_use_erasing

    return adv_image


def save_adv_images(adv_image_tensor: torch.Tensor, path_org: str, out_dir: Path, cluster_num: int):
    folder = os.path.basename(os.path.dirname(path_org))
    name = os.path.basename(path_org)
    name_noext = os.path.splitext(name)[0]
    cluster_dir = out_dir / f"cluster_{cluster_num}" / folder
    ensure_dir(cluster_dir)
    save_path = cluster_dir / (name_noext + ".png")
    torchvision.utils.save_image(adv_image_tensor, save_path)
    return save_path


@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models_lambda_attn_10_std")
def main(cfg: MainConfig):
    seed = getattr(cfg, "seed", 2023)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    time_dir = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{cfg.data.method}_adv_{cfg.data.num_samples}"
    output_path = Path(cfg.data.output) / time_dir
    ensure_dir(output_path)

    code_snapshot_paths = [
        Path("gen_adv.py"),
        Path("surrogates/__init__.py"),
        Path("surrogates/FeatureExtractors/__init__.py"),
        Path("surrogates/FeatureExtractors/Base.py"),
        Path("surrogates/FeatureExtractors/ClipB16.py"),
        Path("surrogates/FeatureExtractors/ClipB32.py"),
        Path("surrogates/FeatureExtractors/ClipLaion.py"),
    ]
    for src_path in code_snapshot_paths:
        dst_path = output_path / src_path
        ensure_dir(dst_path.parent)
        shutil.copy2(src_path, dst_path)

    config_path = Path("config/ensemble_3models_lambda_attn_10_std.yaml")
    shutil.copy2(config_path, output_path / config_path.name)

    transform_fn = transforms.Compose([
        transforms.Resize(cfg.model.input_res, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg.model.input_res),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Lambda(lambda img: to_tensor(img)),
    ])

    clean_data  = ImageFolderWithPaths(cfg.data.cle_data_path, transform=transform_fn)
    target_data = ImageFolderWithPaths(cfg.data.tgt_data_path, transform=transform_fn)

    def sort_key(sample_tuple):
        filename = os.path.basename(sample_tuple[0])
        return int(os.path.splitext(filename)[0])

    clean_data.samples = sorted(clean_data.samples, key=sort_key)
    clean_data.imgs = clean_data.samples

    target_data.samples = sorted(target_data.samples, key=sort_key)
    target_data.imgs = target_data.samples

    num_samples_to_use = cfg.data.num_samples
    total_available = len(clean_data)

    if num_samples_to_use <= 0:
        selected_indices = []
    elif getattr(cfg.data, "sample_mode", "interval") == "first":
        selected_indices = list(range(min(num_samples_to_use, total_available)))
    else:
        interval = total_available // num_samples_to_use if num_samples_to_use > 0 else 1
        selected_indices = list(range(0, total_available, interval))[:num_samples_to_use]

    clean_data = torch.utils.data.Subset(clean_data, selected_indices)
    target_data = torch.utils.data.Subset(target_data, selected_indices)

    data_loader_imagenet = torch.utils.data.DataLoader(clean_data,  batch_size=cfg.data.batch_size, shuffle=False)
    data_loader_target   = torch.utils.data.DataLoader(target_data, batch_size=cfg.data.batch_size, shuffle=False)

    source_crop = transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale) if cfg.model.use_source_crop else torch.nn.Identity()
    target_crop = transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale) if cfg.model.use_target_crop else torch.nn.Identity()

    shared_models = None
    model_cache = {}
    manifest_samples = []

    for i, ((image_org, _, path_org), (image_tgt, _, path_tgt)) in enumerate(zip(data_loader_imagenet, data_loader_target)):
        if cfg.data.batch_size * (i + 1) > cfg.data.num_samples:
            break
        print(f"\nProcessing batch {i+1}")

        batch_size = image_org.shape[0]
        for b in range(batch_size):
            path_b = path_org[b]
            image_org_b = image_org[b:b+1].to(cfg.model.device)
            image_tgt_b = image_tgt[b:b+1].to(cfg.model.device)

            try:
                name_noext = os.path.splitext(os.path.basename(path_b))[0]
                found_tgt_path = None
                for ext in VALID_IMAGE_EXTENSIONS:
                    cand = os.path.join(cfg.data.tgt_data_path, "1", name_noext + ext)
                    if os.path.exists(cand):
                        found_tgt_path = cand
                        break
                if not found_tgt_path:
                    found_tgt_path = path_tgt[b] if isinstance(path_tgt, (list, tuple)) else path_tgt
            except Exception:
                found_tgt_path = path_tgt[b] if isinstance(path_tgt, (list, tuple)) else path_tgt

            adv_paths = {}

            for cluster_num in cfg.dfra_attack.cluster_sequence:
                print(f"--> Generating adversarial image {path_b} with cluster_num={cluster_num}")

                mask_ratio = float(_get_cluster_value(cfg.dfra_attack.mask_ratio_by_cluster, cluster_num, "mask_ratio_by_cluster"))
                erasing_scale = (mask_ratio, mask_ratio)
                lambda_attn = float(_get_cluster_value(cfg.dfra_attack.lambda_attn_by_cluster, cluster_num, "lambda_attn_by_cluster"))
                lambda_rel = float(cfg.dfra_attack.lambda_rel)
                use_random_erasing = bool(cfg.dfra_attack.use_random_erasing)
                use_saliency_mask = bool(cfg.dfra_attack.use_saliency_mask)

                cache_key = (
                    cluster_num,
                    tuple(erasing_scale),
                    float(lambda_attn),
                    float(lambda_rel),
                    bool(use_random_erasing),
                    bool(use_saliency_mask),
                )
                if cache_key not in model_cache:
                    if shared_models is None:
                        shared_models = build_backbone_models(cfg)
                    ensemble_extractor, models, ensemble_loss = get_models_ot_with_cluster(
                        cfg,
                        cluster_num,
                        use_random_erasing=use_random_erasing,
                        erasing_scale=erasing_scale,
                        lambda_attn=lambda_attn,
                        lambda_rel=lambda_rel,
                        use_saliency_mask=use_saliency_mask,
                        models=shared_models,
                    )
                    model_cache[cache_key] = (ensemble_extractor, models, ensemble_loss)
                else:
                    ensemble_extractor, models, ensemble_loss = model_cache[cache_key]

                attack_type = cfg.attack
                attack_fn_map = {
                    "fgsm": fgsm_attack,
                    "mifgsm": mifgsm_attack,
                    "pgd":  pgd_attack,
                }
                attack_fn = attack_fn_map.get(attack_type, pgd_attack)

                adv_image = attack_fn(
                    cfg=cfg,
                    ensemble_extractor=ensemble_extractor,
                    ensemble_loss=ensemble_loss,
                    source_crop=source_crop,
                    target_crop=target_crop,
                    img_index=i,
                    image_org=image_org_b,
                    image_tgt=image_tgt_b,
                )
                adv_save_path = save_adv_images(adv_image[0].cpu(), path_b, output_path, cluster_num)
                adv_paths[str(cluster_num)] = str(adv_save_path)

            manifest_samples.append({
                "original_path": str(Path(path_b).absolute()),
                "target_path": str(Path(found_tgt_path).absolute()),
                "adv_paths": adv_paths,
            })

    manifest = {
        "config": {
            "method": cfg.data.method,
            "num_samples": cfg.data.num_samples,
            "cluster_sequence": cfg.dfra_attack.cluster_sequence,
            "attack_type": cfg.attack,
        },
        "samples": manifest_samples,
    }

    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Adversarial generation complete!")
    print(f"Results saved to: {output_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Next step: python eval_adv.py adv_dir={output_path}")


if __name__ == "__main__":
    main()
