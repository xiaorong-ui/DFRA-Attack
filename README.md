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


## 💖 Acknowledgements

This project is built upon [M-Attack](https://github.com/VILA-Lab/M-Attack) . We thank the authors for their foundational work.


