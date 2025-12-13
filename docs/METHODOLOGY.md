# Methodology

This section describes the theoretical framework and algorithmic approach for Partial Domain Adaptation in agricultural disease detection.

---

## 3.1. Problem Formulation: Partial Domain Adaptation

We address the problem of cross-domain plant disease diagnosis where the source domain (Laboratory) contains a larger label space than the target domain (Field). Formally, let the source domain be $\mathcal{D}_s = \{(x_i^s, y_i^s)\}_{i=1}^{N_s}$ with label space $\mathcal{C}_s$, and the target domain be $\mathcal{D}_t = \{(x_j^t)\}_{j=1}^{N_t}$ with label space $\mathcal{C}_t$.

In agricultural deployment, we frequently encounter the **Partial Domain Adaptation (PDA)** setting where $\mathcal{C}_t \subset \mathcal{C}_s$. Specifically, laboratory datasets often contain a "Healthy" class or irrelevant crop varieties that are absent in disease-specific field outbreaks. Let $\mathcal{C}_{out} = \mathcal{C}_s \setminus \mathcal{C}_t$ denote the "outlier" classes.

Standard Domain Adaptation methods minimize the discrepancy between the marginal distributions $P(X_s)$ and $P(X_t)$. However, in the PDA setting, this forces the target data to align with the entire source distribution, including $\mathcal{C}_{out}$. This leads to **Negative Transfer**, where the model learns to map background noise in the target domain to the specific features of the outlier classes (e.g., mapping soil texture to the "Healthy" label). Our objective is to learn a function $f_\theta$ that maximizes accuracy on $\mathcal{C}_t$ while suppressing activations for $\mathcal{C}_{out}$ using a limited budget of labeled target samples.

---

## 3.2. Dataset Configuration and Semantic Alignment

To empirically validate our framework, we align the **PlantVillage** (Source) and **PlantDoc** (Target) datasets across three distinct crop scenarios (Table II). We implement a canonical mapping strategy to reconcile naming discrepancies between the datasets.

### Crop Scenarios

- **Potato (The PDA Testbed):** This scenario represents the core PDA challenge. The source contains three classes: *Early Blight*, *Late Blight*, and *Healthy*. The target field data contains only *Early* and *Late Blight*; the *Healthy* class is absent ($\mathcal{C}_{out} = \{\text{Healthy}\}$). This setup tests the model's ability to avoid hallucinating "Healthy" labels on noisy field backgrounds.

- **Tomato (Complex Shift):** A high-complexity scenario with 8 shared classes and substantial background clutter, representing a severe domain shift.

- **Pepper (Control Group):** A standard domain adaptation scenario where $\mathcal{C}_s = \mathcal{C}_t$ (2 classes), used to isolate the effects of texture shift from label asymmetry.

### Table II: Dataset Configuration and Label Asymmetry

| Crop | Source Images | Target Pool | Shared Classes ($\mathcal{C}_t$) | Outlier Classes ($\mathcal{C}_{out}$) | Scenario |
|:-----|:-------------:|:-----------:|:--------------------------------:|:-------------------------------------:|:---------|
| **Potato** | 4,304 | 442 | 2 | 1 (Healthy) | **Partial DA** |
| **Tomato** | 25,862 | 1,487 | 8 | 2 | Complex PDA |
| **Pepper** | 4,949 | 264 | 2 | 0 | Standard DA |

---

## 3.3. Edge-Optimized Architecture

Given the constraint of real-time robotic deployment (inference latency $< 100$ ms), we utilize **MobileNetV3-Small** as the primary backbone. It features a lightweight design (~1.5M parameters, 0.06 GFLOPs) utilizing hard-swish activation and Squeeze-and-Excitation (SE) modules. For benchmarking purposes, we compare this against **EfficientNet-B0** (representing heavy CNNs) and **MobileViT-XS** (representing Vision Transformers) to decouple architectural robustness from algorithmic adaptation.

### Architecture Comparison

| Model | Parameters | GFLOPs | Target Latency |
|:------|:----------:|:------:|:--------------:|
| MobileNetV3-Small | 1.5M | 0.06 | < 10 ms |
| EfficientNet-B0 | 5.3M | 0.39 | < 20 ms |
| MobileViT-XS | 2.3M | 0.70 | < 30 ms |

---

## 3.4. Active Label Acquisition Strategy

To minimize annotation costs, we employ an iterative Active Learning (AL) loop. We identify two failure modes in standard AL:

1. **Inefficiency:** Random sampling requires excessive data to capture rare classes.
2. **Cold Start:** Pure uncertainty sampling (e.g., Entropy) diverges in early rounds because the domain-shifted model yields high uncertainty on outliers rather than informative samples.

### Hybrid Warm-Start Strategy

We propose a **Hybrid Warm-Start** strategy. For a label budget $B$ in round $r$:

1. **Round $r=0$ (Warm-Start):** We query $0.3B$ samples via Random selection to approximate the target distribution $P(X_t)$, and $0.7B$ via Entropy sampling to capture decision boundaries.

2. **Rounds $r>0$ (Exploitation):** We transition to pure Entropy sampling:

$$
x^* = \operatorname*{argmax}_{x \in \mathcal{U}} \left( - \sum_{c \in \mathcal{C}_s} P(y=c|x) \log P(y=c|x) \right)
$$

This hybrid approach prevents the model from overfitting to source-specific biases in the initial adaptation phase.

### Algorithm 1: Hybrid Active Learning

```
Input: Source model f_θ, Target pool U, Budget B, Rounds R
Output: Adapted model f_θ'

for r = 0 to R-1 do:
    if r == 0 then:
        S_random ← RandomSample(U, 0.3 × B)
        S_entropy ← EntropySample(U \ S_random, 0.7 × B)
        S ← S_random ∪ S_entropy
    else:
        S ← EntropySample(U, B)
    
    L ← L ∪ S          // Add to labeled set
    U ← U \ S          // Remove from pool
    f_θ ← FineTune(f_θ, L)
    
return f_θ
```

---

## 3.5. Semi-Supervised Consistency Regularization

To leverage the unlabeled portion of the target pool ($\mathcal{U}$) and suppress negative transfer, we integrate **FixMatch**. This component serves as a "domain filter" for the PDA problem.

### FixMatch Formulation

For an unlabeled image $u_b$, we generate a weakly augmented view $\alpha(u_b)$ (flip/shift) and a strongly augmented view $\mathcal{A}(u_b)$ (RandAugment). The objective function is:

$$
\mathcal{L}_u = \frac{1}{\mu B} \sum_{b=1}^{\mu B} \mathbb{1}(\max(q_b) \ge \tau) \cdot H(\hat{y}_b, p_m(\mathcal{A}(u_b)))
$$

where:
- $q_b = p_m(y | \alpha(u_b))$ is the prediction on the weak view
- $\hat{y}_b = \operatorname{argmax}(q_b)$ is the pseudo-label
- $\tau$ is the confidence threshold
- $H(\cdot, \cdot)$ is the cross-entropy loss

### Mechanism for PDA

We set a strict confidence threshold $\tau = 0.95$. In a PDA setting, target samples that visually resemble source outliers (e.g., a background looking like a "Healthy" leaf) typically yield ambiguous, low-confidence predictions from the source model. Consequently, $\max(q_b) < \tau$, and these samples are **masked out** of the loss calculation.

Conversely, samples belonging to the shared target classes ($\mathcal{C}_t$) exhibit higher semantic consistency. Thus, FixMatch implicitly re-weights the optimization landscape to focus on the shared classes, effectively pruning the $\mathcal{C}_{out}$ manifold without requiring explicit prior knowledge of the target label space.

### Combined Training Objective

The final loss function combines supervised and unsupervised components:

$$
\mathcal{L}_{total} = \mathcal{L}_s + \lambda_u \mathcal{L}_u
$$

where:
- $\mathcal{L}_s$ is the supervised cross-entropy loss on labeled target samples
- $\mathcal{L}_u$ is the FixMatch consistency loss on unlabeled samples
- $\lambda_u$ is the unsupervised loss weight (default: 1.0)

---

## 3.6. Training Protocol

### Baseline Training (Phase 1)
- **Dataset:** PlantVillage (Source) only
- **Split:** 80% Train / 20% Validation
- **Epochs:** 10
- **Optimizer:** Adam with learning rate $10^{-3}$
- **Augmentation:** Random flip, rotation (±15°), color jitter

### Active Learning Adaptation (Phases 3-5)
- **Initial Model:** Pretrained baseline from Phase 1
- **Budget:** 10 samples per round
- **Rounds:** 5 (total: 50 labeled target samples)
- **Fine-tuning:** 5-15 epochs per round with learning rate $10^{-4}$
- **FixMatch:** Enabled in Phase 4-5 with $\tau = 0.95$

### Evaluation Protocol
- **Test Set:** 20% of PlantDoc (held out, never used for training)
- **Metrics:** Top-1 Accuracy, Per-class Precision/Recall, Confusion Matrix
- **Key Indicator:** Predictions for $\mathcal{C}_{out}$ classes (should be 0 in PDA scenario)

