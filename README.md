<h3 align="center">⚔️ Adversarial Attacks against Closed-Source MLLMs via Defense-aware Feature Relation Alignment</h3>
<p align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=xiaorong-ui.DFRA-Attack" alt="Visitor Badge" />
  <img src="https://img.shields.io/github/stars/xiaorong-ui/DFRA-Attack?style=social" alt="GitHub Stars" />
  <img alt="License Badge" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" />
</p>

> **DFRA-Attack** enhances adversarial transferability in multimodal large language models by optimizing global and local feature alignments using cosine similarity and optimal transport-based methods.

## 💥 News
- **[2025-05-07]** Code release: Two-stage pipeline with parallel processing support! 🚀

## 📋 Overview

DFRA-Attack is a Python framework for generating transferable adversarial examples targeting vision-language models (VLMs). It features:

- **Two-stage workflow**: Separate image generation and evaluation phases
- **Multi-backbone ensemble**: CLIP B16, B32, and Laion feature extractors
- **Parallel processing**: Support for concurrent GPU execution (5-8 processes)
- **Configurable parameters**: Mask ratio, attention weights, relation loss scaling
- **Automated tracking**: Manifest.json for reproducibility

## 💻 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
conda env create -f environment.yml
```

## 🛰️ Quick Start

### Method 1: End-to-End Pipeline (Single Process)
```bash
python DFRAAttack.py
```

### Method 2: Two-Stage Pipeline (Recommended for Large-Scale Experiments)

**Stage 1: Generate Adversarial Images (GPU-intensive)**
```bash
python generate_adv_stages.py \
  --config config/ensemble_3models_50_random_erasing.yaml \
  --output-root ./results_1000 \
  --sample-start 0 --sample-end 200 \
  --stage3-erasing 0.1 0.1 --stage5-erasing 0.1 0.1 \
  foa_attack.lambda_attn_by_cluster={3:0.15,5:0.15} \
  foa_attack.lambda_rel=0.5 \
  model.device=cuda:0
```

**Stage 2: Evaluate Generated Images (API-based)**
```bash
python eval_adv.py \
  manifest_path=results_1000/TIMESTAMP/attack_manifest.json \
  model.device=cuda:0
```

For parallel execution with multiple GPUs, see [SPLIT_WORKFLOW.md](SPLIT_WORKFLOW.md).

## 🔧 Configuration

Edit `config/*.yaml` files to customize:
- `mask_ratio_by_cluster`: Perturbation magnitude per stage
- `lambda_attn_by_cluster`: Attention loss weight
- `lambda_rel`: Relation loss weight
- `optim.steps`: Optimization iterations (default: 300)
- `optim.alpha`: Step size

## 📊 Parallel Processing Example

Run 8 concurrent processes across 4 GPUs:

```bash
# Process 1-8 with different sample ranges and GPU assignments
for i in {0..7}; do
  nohup python generate_adv_stages.py \
    --sample-start $((i*125)) --sample-end $(((i+1)*125)) \
    model.device=cuda:$((i%4)) \
    > gen_adv_p$((i+1)).log 2>&1 &
done
```

## 📁 Output Structure

```
results_1000/
└── TIMESTAMP_DFRA_attack_only_1000/
    ├── stage1/          # Stage 3 generated images
    │   ├── 0/
    │   │   ├── 0.png
    │   │   └── ...
    │   └── ...
    ├── stage2/          # Stage 5 generated images
    │   └── ...
    ├── attack_manifest.json  # Metadata and tracking
    └── (code snapshots)
```

## 🎯 Attack Methods

Supported optimization algorithms:
- **FGSM**: Fast Gradient Sign Method
- **MI-FGSM**: Momentum Iterative FGSM
- **PGD**: Projected Gradient Descent

## 💖 Acknowledgements

This project is built upon [M-Attack](https://github.com/VILA-Lab/M-Attack) and [FOA-Attack](https://github.com/PKU-YuanGroup/FOA-Attack). We thank the authors for their foundational work.

## 📝 Citation

If you use DFRA-Attack in your research, please cite:

```bibtex
@article{dfra-attack-2025,
  title={DFRA-Attack: Defense-aware Feature Relation Alignment for Adversarial Attacks},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## 📄 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
