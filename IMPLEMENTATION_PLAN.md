This is the **Implementation Specification** for your experimental protocol. This is your blueprint. Do not deviate from this without a specific scientific reason.

We are simulating a **Partial Domain Adaptation (PDA)** scenario where:
1.  **Source Domain ($\mathcal{D}_s$):** Laboratory conditions (controlled, clean).
2.  **Target Domain ($\mathcal{D}_t$):** Field conditions (noisy, complex background).
3.  **The Shift:** The Target domain contains "Phantom Classes" (Weeds, Soil, Mulch) that look nothing like the Source's "Healthy" class, yet the model tries to map them there.

---

### 1. The Dataset Configuration ("The Valley of Death")

You will use the **Tomato** subset of two standard datasets to construct the PDA scenario.

#### **A. Source Domain: PlantVillage (Laboratory)**
*   **Characteristics:** Single leaf, plain gray/black background, perfect lighting.
*   **Classes ($\mathcal{Y}_s$):** All 10 Tomato classes (9 Diseases + 1 Healthy).
*   **Role:** Fully labeled training data.

#### **B. Target Domain: PlantDoc (Field/Wild)**
*   **Characteristics:** Multiple leaves, chaotic background (soil, sticks, human hands), variable lighting.
*   **Classes ($\mathcal{Y}_t$):** A subset of 6 common classes + "Background/Noise".
*   **Role:** Unlabeled (mostly), with a small Active Learning budget ($N=50$).

#### **The PDA Setup (Crucial)**
We define the label spaces to force **Negative Transfer**:
*   **Shared Classes ($\mathcal{Y}_s \cap \mathcal{Y}_t$):** *Early Blight, Late Blight, Septoria Leaf Spot, Mosaic Virus, Yellow Leaf Curl, Healthy.*
*   **Source-Only Classes (Outliers):** *Target Spot, Spider Mites, Bacterial Spot.*
*   **The Trap:** PlantDoc images often contain huge patches of **Soil/Mulch**. A standard model trained on PlantVillage (where "Healthy" = Green Leaf on Black Background) will effectively guess on these soil patches. *Your goal is to reject these.*

---

### 2. Model Architectures (The "Edge" Constraint)

You must train and evaluate these specific backbones.

| Role | Architecture | Parameters | GFLOPs | Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Main (Ours)** | **MobileNetV3-Small** | **1.5M** | **0.06** | **Target deployment hardware (Jetson Nano).** |
| Baseline (SSL) | WideResNet-28-2 | 1.5M | ~0.5 | Standard backbone for FixMatch papers. |
| Baseline (UDA) | ResNet-50 | 25M | 4.1 | Standard backbone for PADA/Domain Adaptation. |
| Upper Bound | Swin-Tiny (ViT) | 28M | 4.5 | Represents "Heavy" Modern Transformers. |

---

### 3. Algorithms & Baselines (The "Competitors")

You need to implement these four training pipelines.

#### **A. Source-Only (Lower Bound)**
*   **Logic:** Train MobileNetV3 on PlantVillage (Source). Test directly on PlantDoc (Target).
*   **Expected Result:** High accuracy on lab test set (>99%), catastrophic failure on field test set (~20-25%).

#### **B. PADA: Partial Adversarial Domain Adaptation (The "Classic" Rival)**
*   **Logic:** Adversarial learning. A Generator ($G$) tries to fool a Domain Discriminator ($D$).
*   **Key Mechanism:** Class-level weighting. PADA calculates the contribution of each source class to the target domain and down-weights the "Source-Only" classes (Spider Mites, etc.) to prevent negative transfer.
*   **Why here?** To prove that adversarial methods are too heavy/unstable for 50 samples.

#### **C. MW-FixMatch (The "Dangerous" Rival)**
*   **Logic:** Standard FixMatch (Consistency Regularization) + Meta-Weighting (MW).
*   **Key Mechanism:** It uses a validation set (meta-data) to learn weights for the loss function, aggressively up-weighting "rare" examples.
*   **Failure Mode:** It will mistake "Background/Soil" for a "Rare Class" and up-weight it (False Positive Amplification).

#### **D. Hybrid Warm-Start FixMatch (Ours)**
*   **Phase 1: Active Learning (The "Warm Start").**
    *   **Algorithm:** K-Means Clustering on the target feature space.
    *   **Process:** Embed unlabeled target $\to$ Cluster into $K=50$ $\to$ Select nearest to centroid.
    *   **Result:** A balanced initial set $\mathcal{L}_{warm}$ that covers disease AND background.
*   **Phase 2: FixMatch Adaptation.**
    *   **Logic:** Fine-tune on $\mathcal{L}_{warm}$, then run FixMatch on the rest of $\mathcal{U}$.
    *   **Key Difference:** Because we "anchored" the background clusters in Phase 1, FixMatch will correctly pseudo-label background as "Background" (or Low Confidence), rather than hallucinating "Disease."

---

### 4. Implementation Details (Hyperparameters)

Do not guess these. Use standard values to ensure reproducibility.

*   **Framework:** PyTorch.
*   **Image Size:** $224 \times 224$ (Standard).
*   **Optimizer:** SGD with Momentum (0.9).
*   **Learning Rate:** 0.03 (decay by 0.1 at epochs 30, 50).
*   **Batch Size:** 64 (Source), 32 (Target Labeled), 32 (Target Unlabeled).
*   **FixMatch Params:**
    *   Threshold ($\tau$): **0.95** (High threshold is critical for avoiding noise).
    *   Unlabeled Ratio ($\mu$): **7** (7 unlabeled images for every 1 labeled image).
*   **Active Learning Budget:** $N = 50$ (Strict).

---

### 5. Code Structure (Blueprint)

Your repository should look like this. Start coding `dataset.py` and `sampler.py` immediately.

```text
/project_root
├── main.py                   # Argument parser and training loop
├── config.py                 # Hyperparams (LR, Batch Size, Budget)
├── /src
│   ├── /data
│   │   ├── dataset.py        # Source & Target Dataloaders
│   │   └── sampler.py        # Active Selection Logic
│   ├── /models
│   │   ├── mobilenetv3.py    # Your edge model
│   │   └── resnet.py         # For PADA/MW-FixMatch baselines
│   ├── /algorithms
│   │   ├── fixmatch.py       # Standard SSL logic
│   │   ├── pada.py           # Adversarial logic
│   │   └── active_selection.py # <--- YOUR CORE CONTRIBUTION
│   └── /utils
│       └── metrics.py        # Accuracy, F1, and "Healthy False Positive Rate"
```

### 6. The "Golden Metric"

In your results, you must report **"Healthy False Positive Rate" (FPR)** on the Target domain.
*   **Definition:** Of all the "Background/Soil" images in PlantDoc, what percentage did the model classify as "Disease"?
*   **Hypothesis:**
    *   MW-FixMatch FPR: High (~40%). (Overfitting to noise).
    *   Ours FPR: Low (<5%). (Correctly filtered).

**Go build the `dataset.py` loaders first. If you can't load the data, you can't do the science.**
