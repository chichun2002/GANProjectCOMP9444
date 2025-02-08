# Unpaired Image to Image Translation on Video Game Graphics

---

| Melanie Daixing |
| Tristan Fischer |
| Calvin Liu      |
| Leon Xu         |
| Guozhi Zhao     |

---

## 1 Introduction

### 1.1 Background

The visual style of older games can often appear dated and low in quality due to the technical limits of computer hardware being unable to create high-end visual scenery and graphics at the time. As a result, many older games often feature pixelated textures, low-resolution assets, simplified visual effects and a dimmer, less vibrant design.

Traditional methods of improving the graphics quality of older games involve completely redesigning the game, which can be extremely time consuming and expensive. Our project attempts to use an approach that utilises AI that enhances the visual appeal of older games without completely redesigning them.

### 1.2 Motivation

With rising expectations for visual immersion, we aim to transfer modern game styles to older games of similar genres while preserving thematic coherence.

### 1.3 Problem Statement

Our objective for this project is to enhance the visual aesthetics of computer games designed in the early 2000s by using style transfer methods on more modern games designed around 20 years later. We will be analysing the accuracy of the results obtained from using different methods on the same dataset. Specifically, we have chosen to enhance the visuals of Counter Strike 1.6 (released in 2003) by using style transfer methods on Valorant (released in 2020).

---

## 2 Dataset

### 2.1 Data Information

**Dataset Link**: [COMP9444 Dataset](https://unsw-my.sharepoint.com/personal/z5310974_ad_unsw_edu_au/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fz5310974%5Fad%5Funsw%5Fedu%5Fau%2FDocuments%2FDataset%20COMP9444)

- **Sources**: 
  - *Counter Strike 1.6*: Prerecorded gameplay converted to image sequences.
  - *Valorant*: Custom gameplay recordings (combat/environment) converted to images.
- **Split**: 4,000 training and 4,000 testing images per game.

### 2.2 Data Parameters

- **CycleGAN Baseline**: 100 epochs (original paper), scaled to 20 epochs for our larger dataset.
- **Batch Sizes**: 4 for most models; 1 for UNSB due to memory constraints.

---

## 3 Models

### 3.1 Generative Adversarial Networks (GANs)

GANs involve a generator and discriminator competing to produce realistic data. We explore variants for image translation.

### 3.2 CycleGAN

**Paper**: [Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)  
**Code**: [pytorch-CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

- **Architecture**: 28.3M parameters (2 generators, 2 discriminators).
- **Cycle Consistency**: Enforces reversibility between domains (e.g., `F(G(X)) ≈ X`).

#### Training
```python
python train.py --dataroot "path/to/dataset" --name game_transfer_cycle_gan --model cycle_gan --batch_size 4 --n_epochs 20 --n_epochs_decay 20
```

#### Testing
```python
python test.py --dataroot "path/to/dataset" --name game_transfer_cycle_gan --model cycle_gan --batch_size 4 --num_test 4000 --phase test
```

---

### 3.3 Contrastive Unpaired Translation (CUT) GAN

**Paper**: [Contrastive Learning for Unpaired Translation](https://arxiv.org/pdf/2007.15651)  
**Code**: [CUT-GAN](https://github.com/taesungp/contrastive-unpaired-translation)

- **Architecture**: 14.7M parameters.
- **Feature Matching**: Maximizes mutual information between input/output patches.

#### Training
```python
python train.py --dataroot "path/to/dataset" --name CUT_Model --CUT_mode CUT --batch_size 4 --n_epochs 20 --n_epochs_decay 20
```

#### Testing
```python
python test.py --dataroot "path/to/dataset" --name CUT_Model --CUT_mode CUT --phase test --batch_size 4 --num_test 4000
```

---

### 3.4 DECENT (Density Changing Regularization)

**Paper**: [DECENT](https://openreview.net/pdf?id=RNZ8JOmNaV4)  
**Code**: [DECENT](https://github.com/Mid-Push/Decent)

- **Architecture**: 22.7M parameters.
- **Density Estimators**: Penalizes patch density variances during translation.

#### Training
```python
python train.py --dataroot="path/to/dataset" --batch_size 4 --n_epochs 20 --n_epochs_decay 20 --name DECENT --lambda_var 0.01 --var_all --flow_blocks 1 --flow_lr 0.001 --flow_type bnaf
```

---

### 3.5 UNSB (Neural Schrödinger Bridge)

**Paper**: [UNSB](https://arxiv.org/abs/2305.15086)  
**Code**: [UNSB](https://github.com/cyclomon/UNSB)

- **Architecture**: 21.5M parameters.
- **Diffusion Models**: Combines Schrödinger Bridge with GANs for high-quality synthesis.

#### Training
```python
python train.py --dataroot="path/to/dataset" --batch_size 1 --n_epochs 20 --n_epochs_decay 20 --name UNSB --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0
```

---

## 4 Results

### 4.1 FID/KID Scores (Lower is Better)

| Model    | FID ↓   | KID ↓   |
|----------|---------|---------|
| CycleGAN | 125.2   | 0.107   |
| UNSB     | 169.8   | 0.165   |
| CUT      | 180.8   | 0.178   |
| DECENT   | 211.3   | 0.216   |

**Metrics**:
- **FID**: Measures distribution similarity between real/fake images.
- **KID**: Kernel-based variant for smaller sample sizes.

---

### 4.2 FVD Scores (Lower is Better)

| Model    | FVD2048_16F ↓ | FVD2048_128F ↓ | FVD2048_128F_ss8F ↓ |
|----------|---------------|----------------|---------------------|
| CycleGAN | 2208.7        | 2152.8         | 1978.6              |
| UNSB     | 3641.5        | 2409.5         | 2407.4              |
| CUT      | 2528.2        | 4309.3         | 1306.9              |
| DECENT   | 2990.9        | 3388.4         | 2183.4              |

**Metric**: FVD evaluates temporal coherence in generated videos.

---

### 4.3 Training Time

| Model    | Training Time (min/epoch) |
|----------|---------------------------|
| CycleGAN | 10                        |
| UNSB     | 33                        |
| CUT      | 12                        |
| DECENT   | 7.5                       |

---

## 5 Analysis

### 5.1 Results Summary
- **CycleGAN** outperformed others in FID/KID and maintained texture details.
- **CUT/UNSB** showed moderate performance but struggled with gun models.
- **DECENT** prioritized style over structure, leading to artifacts.

### 5.2 Limitations
- Dataset inconsistencies (e.g., minimap presence in Valorant).
- Short training cycles due to resource constraints.

### 5.3 Future Work
- Expand model variety and real-time translation capabilities.
- Improve dataset consistency (e.g., disable minimap, standardize weapons).

---

## 6 References

- Zhu, J.Y., Park, T., Isola, P. and Efros, A.A., 2017. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision (pp. 2223-2232)*.
- Park, T., Efros, A.A., Zhang, R. and Zhu, J.Y., 2020. Contrastive learning for unpaired image-to-image translation. In Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part IX 16 (pp. 319-345). Springer International Publishing*.
- Xie, S., Ho, Q. and Zhang, K., 2022. Unsupervised image-to-image translation with density changing regularization. Advances in Neural Information Processing Systems, 35, pp.28545-28558*.
- Kim, B., Kwon, G., Kim, K. and Ye, J.C., 2023. Unpaired Image-to-Image Translation via Neural Schrödinger Bridge. arXiv preprint arXiv:2305.15086. *Betzalel, E., Penso, C., Navon, A. and Fetaya, E., 2022. A study on the evaluation of generative models. arXiv preprint arXiv:2206.10935*.
- Lucic, M., Kurach, K., Michalski, M., Gelly, S. and Bousquet, O., 2018. Are gans created equal? a large-scale study. Advances in neural information processing systems, 31*.
- Unterthiner, T., Van Steenkiste, S., Kurach, K., Marinier, R., Michalski, M. and Gelly, S., 2018. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717*.
```
