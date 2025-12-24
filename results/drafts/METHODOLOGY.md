Here is the complete **Methodology** section. I have refined it to strictly align with the data artifacts (crop configs, splits) and the theoretical framework (Negative Transfer) we established.

Copy and paste this directly into **Section III** of your manuscript.

***

# III. METHODOLOGY

## A. Problem Formulation: Partial Domain Adaptation
We address the problem of cross-domain plant disease diagnosis where the label space of the source domain (Laboratory) subsumes that of the target domain (Field). Formally, let the source domain be $\mathcal{D}_s = \{(x_i^s, y_i^s)\}_{i=1}^{N_s}$ with label space $\mathcal{Y}_s$, and the target domain be $\mathcal{D}_t = \{(x_j^t)\}_{j=1}^{N_t}$ with label space $\mathcal{Y}_t$.

In agricultural deployment, we encounter the **Partial Domain Adaptation (PDA)** setting where $\mathcal{Y}_t \subset \mathcal{Y}_s$. Specifically, laboratory datasets contain "outlier" classes (denoted as $\mathcal{Y}_{out} = \mathcal{Y}_s \setminus \mathcal{Y}_t$) such as "Healthy" leaves or irrelevant crop varieties that are absent in disease-specific field outbreaks. Standard Domain Adaptation methods minimize the discrepancy between marginal distributions $P(X_s)$ and $P(X_t)$. However, in the PDA setting, this forces the target data to align with the entire source distribution, including $\mathcal{Y}_{out}$. This leads to **Negative Transfer**, where the model minimizes loss by mapping background noise in the target domain to the specific features of the outlier classes (e.g., mapping soil texture to the "Healthy" label).

Our objective is to learn a function $f_\theta$ that maximizes accuracy on $\mathcal{Y}_t$ while suppressing activations for $\mathcal{Y}_{out}$, utilizing a limited budget of labeled target samples acquired via Active Learning.

## B. Dataset Configuration and Semantic Alignment
To empirically validate our framework, we align the **PlantVillage** (Source) and **PlantDoc** (Target) datasets across three distinct crop scenarios. We implemented a canonical mapping strategy to reconcile naming discrepancies (e.g., *Tomato Yellow Leaf Curl Virus* vs. *Yellow Virus*) and explicitly defined the PDA outlier sets (Table II).

1.  **Potato (The PDA Testbed):** This scenario represents the core PDA challenge. The source contains three classes: *Early Blight*, *Late Blight*, and *Healthy*. The target field data contains only *Early* and *Late Blight*; the *Healthy* class is structurally absent. This allows us to quantify negative transfer by measuring False Positive predictions for the "Healthy" class.
2.  **Tomato (Complex Shift):** A high-complexity scenario with 9 canonical classes. While mostly symmetric, the target domain suffers from extreme class imbalance and visual clutter, representing a "Soft" PDA scenario where rare classes effectively act as outliers.
3.  **Pepper (Control Group):** A standard domain adaptation scenario where $\mathcal{Y}_s = \mathcal{Y}_t$ (2 classes: Bacterial Spot, Healthy), used to isolate the effects of texture shift from label asymmetry.

**[INSERT TABLE II HERE]**
*Caption: Table II. Dataset Configuration. The Potato scenario is a strict Partial Domain Adaptation problem where the 'Healthy' class exists in the Source but not the Target.*

## C. Edge-Optimized Architecture
Given the constraint of real-time robotic deployment (inference latency $<100$ ms), we utilize **MobileNetV3-Small** as the primary backbone. This architecture features a lightweight design (~1.5M parameters, 0.06 GFLOPs) utilizing hard-swish activation and Squeeze-and-Excitation (SE) modules to maximize feature extraction per unit of compute. For benchmarking, we compare this against **EfficientNet-B0** (representing heavy CNNs) and **MobileViT-XS** (representing Vision Transformers) to decouple architectural robustness from algorithmic adaptation.

## D. Hybrid Warm-Start Active Learning
To minimize annotation costs, we employ an iterative Active Learning (AL) loop. We identify two failure modes in standard AL: (1) **Inefficiency:** Random sampling requires excessive data to capture rare classes; (2) **Cold Start:** Pure uncertainty sampling (e.g., Entropy) performs poorly in early rounds because the domain-shifted model yields high uncertainty on outliers (background noise) rather than informative samples [14].

We propose a **Hybrid Warm-Start** strategy. For a label budget $B$ in round $r$:
1.  **Round $r=0$ (Exploration):** We query samples using a ratio of **30% Random / 70% Entropy**. The random component ensures the initial batch approximates the true target distribution $P(X_t)$ regardless of model bias, breaking the Cold Start.
2.  **Rounds $r>0$ (Exploitation):** We transition to pure uncertainty sampling using Shannon Entropy:
    $$ x^* = \operatorname*{argmax}_{x \in \mathcal{U}} \left( - \sum_{c \in \mathcal{Y}_s} P(y=c|x) \log P(y=c|x) \right) $$
This hybrid approach stabilizes the adaptation trajectory before the model is fully calibrated.

## E. Semi-Supervised Consistency Regularization (FixMatch)
To leverage the unlabeled portion of the target pool ($\mathcal{U}$) and suppress negative transfer, we integrate **FixMatch**. This component functions as a "semantic filter" for the PDA problem.

For an unlabeled image $u_b$, we generate a weakly augmented view $\alpha(u_b)$ (flip/shift) and a strongly augmented view $\mathcal{A}(u_b)$ (RandAugment). The consistency loss is minimized only when the model is confident:
$$ \mathcal{L}_u = \frac{1}{\mu B} \sum_{b=1}^{\mu B} \mathbb{1}(\max(q_b) \ge \tau) \cdot H(\hat{y}_b, p_m(\mathcal{A}(u_b))) $$
where $q_b = p_m(y | \alpha(u_b))$ is the prediction on the weak view, $\hat{y}_b$ is the pseudo-label, and $\tau$ is the confidence threshold.

**Mechanism for PDA:** We set a strict confidence threshold $\tau = 0.95$. In a PDA setting, target samples that visually resemble source outliers (e.g., a soil patch resembling a "Healthy" leaf) typically yield ambiguous predictions from the source model. Consequently, $\max(q_b) < \tau$, and these samples are masked out of the loss calculation. Conversely, samples belonging to shared target classes ($\mathcal{Y}_t$) exhibit higher semantic consistency. Thus, FixMatch implicitly re-weights the optimization landscape to focus on the shared classes, effectively pruning the outlier manifold without requiring explicit prior knowledge of the target label space.

**[INSERT FIGURE 1 HERE]**
*Caption: Fig. 1. System Overview. The framework initializes with a Source Model (Left), adapts via Hybrid Active Learning (Center), and refines utilizing unlabeled data via FixMatch (Right).*