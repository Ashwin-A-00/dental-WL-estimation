# 🦷 Automated Dental Working Length Estimation

> **Deep learning pipeline for crown & apex landmark detection in periapical radiographs**  
> UNet · EfficientNet-B3 · DenPAR · Heatmap Regression · Hungarian Matching

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)](https://pytorch.org)
[![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF?logo=kaggle)](https://kaggle.com)
[![Status](https://img.shields.io/badge/Results-13%2F13%20Normal-brightgreen)](#results)

![Working Length Estimation](https://raw.githubusercontent.com/Ashwin-A-00/dental-WL-estimation/main/Results/Working%20length%20estimation.png)

---

## What it does

Endodontic treatment requires precise working length (WL) measurement — the distance from the crown tip to the root apex. This pipeline **automates that measurement** from a periapical X-ray using keypoint detection and clinical-scale calibration, replacing manual ruler estimation.

```
Periapical X-ray  →  CLAHE preprocessing  →  UNet heatmaps  →  Peak detection  →  WL (mm)
```

---

## Pipeline

| Stage | Detail |
|---|---|
| **Preprocessing** | CLAHE contrast enhancement + Gaussian denoise, resized to 512×512 |
| **Model** | UNet with EfficientNet-B3 encoder (ImageNet pretrained) |
| **Output** | 2-channel Gaussian heatmap — channel 0: crown, channel 1: apex |
| **Pretrain** | DenPAR dataset (Training + Validation splits) |
| **Fine-tune** | Custom teacher dataset; 180° flip augmentation to correct X-ray orientation |
| **Matching** | Hungarian algorithm for crown–apex pairing in multi-tooth images |
| **WL Calc** | Euclidean pixel distance × 21.5 mm / 420 px calibration factor |

---

## Results

Evaluated on **5 periapical radiographs** (13 teeth total).  
All predictions fall within the **clinically accepted range of 18–26 mm**.

| X-ray Image | Tooth | Working Length (mm) | Clinical Status |
|---|:---:|:---:|:---:|
| Image 1 | 1 | 19.5 | ✅ Normal |
| | 2 | 21.0 | ✅ Normal |
| | 3 | 19.9 | ✅ Normal |
| Image 2 | 1 | 21.7 | ✅ Normal |
| | 2 | 23.8 | ✅ Normal |
| | 3 | 20.7 | ✅ Normal |
| Image 3 | 1 | 25.3 | ✅ Normal |
| | 2 | 24.7 | ✅ Normal |
| Image 4 | 1 | 22.0 | ✅ Normal |
| | 2 | 22.0 | ✅ Normal |
| Image 5 | 1 | 19.4 | ✅ Normal |
| | 2 | 19.5 | ✅ Normal |
| | 3 | 19.9 | ✅ Normal |

**Mean WL: 21.5 mm · Range: 19.4 – 25.3 mm · 13/13 within clinical limits**

---

## Repo Structure

```
dental-wl-estimation/
├── notebook/
│   └── apex-and-crown-detection.ipynb   # full pipeline
├── results/
│   ├── images/
│   │   ├── sample_output_wl.png         # WL overlaid on X-rays
│   │   └── sample_landmarks.png         # crown & apex markers
│   └── wl_results_final.csv             # per-tooth predictions
├── report/
│   └── working_length_estimation_report.pdf
├── .gitignore
└── README.md
```

---

## Setup

```bash
pip install segmentation-models-pytorch albumentations torch torchvision opencv-python scipy pandas
```

> **Weights:** Pretrained model weights are hosted on Kaggle at `morestorage76/wl-weights`.  
> Download and place at `weights/working_length_final.pth` before running inference.

---

## Key Design Choices

- **180° flip** — DenPAR has apex at bottom; teacher X-rays have apex at top. Flipping teacher images + mirroring annotations aligns the two distributions.
- **Weighted heatmap loss** — 10× penalty near keypoint center to handle extreme class imbalance between foreground and background pixels.
- **Frozen encoder during fine-tune** — preserves DenPAR-learned features; only decoder adapts to teacher data.

---

## Limitations & Next Steps

- [ ] Ground truth WL measurements needed for quantitative eval (MAE, ±2 mm accuracy)  
- [ ] Expand teacher dataset beyond 5 images  
- [ ] Canal segmentation for full root tracing  
- [ ] FIBO ontology integration for structured dental knowledge representation  

---

## Research Context

Part of an ongoing research internship at **NIT Calicut** under **Dr. Pournami P.N.**,  
focused on automated dental image analysis using deep learning.

---

*Built with PyTorch · segmentation-models-pytorch · DenPAR · OpenCV*
