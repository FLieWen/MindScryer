# MindScryer: High-Quality EEG-to-Image Reconstruction via Dual-Branch Feature Extraction and Cascaded Conditional Diffusion

**Abstract:** Reconstructing visual stimuli from EEG signals is a challenging problem at the intersection of neuroscience and computer vision. We propose MindScryer, a novel framework that leverages dual-branch feature extraction to capture both temporal and spectral characteristics of EEG signals, combined with cascaded conditional diffusion models for high-quality image reconstruction. Our approach introduces a self-supervised paradigm for learning EEG time-domain representations, supplements them with frequency-domain features, and employs semantic interpolation for fine-grained cross-modal alignment between EEG embeddings and CLIP semantic space. Using only a fraction of the training data required by prior works, MindScryer achieves state-of-the-art performance in both semantic fidelity and generation quality.

---

## Requirements

- **Python** 3.10+ (tested on 3.10.12)
- **PyTorch** 2.5.1+ with CUDA support
- **GPU**: NVIDIA A100 80GB recommended
- **OS**: Linux (tested on Ubuntu 22.04)

```bash
pip install -r requirements.txt
```

---

## Setup

### 1. Directory Structure

```bash
python create_path.py
```

### 2. Required Resources

| Resource | Expected Path |
|----------|---------------|
| [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14) | `clip-vit-large-patch14/` |
| [BLIP-2 OPT-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) | `blip2-opt-2.7b/` |
| [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) | `pretrained_model/v1-5-pruned-emaonly.ckpt` |
| [EEG-Image Dataset](https://tinyurl.com/eeg-visual-classification) | `data/EEG/` |
| [ImageNet Subset](https://drive.google.com/file/d/1k3Psdqhl0Saiol4Yauy6eCQK6_-Em05R/view) | `data/image/` |

### 3. Precompute Embeddings

```bash
python imageBLIPtoCLIP.py     # BLIP-2 captions → CLIP text embeddings
python imageLabeltoCLIP.py    # ImageNet labels → CLIP text embeddings
```

---

## Training Pipeline

All stages use the `--train_stage` flag:

```bash
# Stage 1: Train frequency encoder
python train_freqencoder.py

# Stage 2: Pretrain time encoder (self-supervised)
python main.py --train_stage pretrain

# Stage 3: Finetune time encoder (classification)
python main.py --train_stage finetune

# Stage 4: Integrate time + frequency models
python main.py --train_stage finetune_timefreq

# Stage 5: Cross-modal EEG-CLIP alignment
python main.py --train_stage finetune_CLIP

# Export alignment results
python main.py --train_stage test
```

---

## Image Reconstruction

```bash
python cascade_diffusion.py
```

Results saved to `picture-gene/`.

---

## Evaluation

```bash
python evaluate_FID.py      # Fréchet Inception Distance
python evaluate_IS.py       # Inception Score
python evaluate_GA.py       # Generation Accuracy (n-way top-k)
```

---

## Project Structure

```
MindScryer/
├── main.py                       # Training entry point
├── args.py                       # Configuration & CLI
├── process.py                    # Trainer (all training stages)
├── loss.py                       # Loss functions
├── dataset.py                    # PyTorch Dataset classes
├── datautils.py                  # EEG data loading
├── cascade_diffusion.py          # Cascaded diffusion reconstruction
├── train_freqencoder.py          # Frequency encoder training
├── classification.py             # Classifier utilities
├── create_path.py                # Directory setup
├── imageBLIPtoCLIP.py            # BLIP-2 → CLIP preprocessing
├── imageLabeltoCLIP.py           # Label → CLIP preprocessing
├── evaluate_FID.py               # FID evaluation
├── evaluate_IS.py                # IS evaluation
├── evaluate_GA.py                # GA evaluation
├── model/
│   ├── MindScryerModels.py       # Core models
│   ├── MindScryerModels_test.py  # Extended models (EEGNet, etc.)
│   ├── layers.py                 # Transformer components
│   └── mlp_classifier.py         # MLP classifier
├── dc_ldm/                       # Diffusion model framework
└── requirements.txt
```

---

## License

MIT License
