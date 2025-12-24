Here is the full draft for **Section 3.2: The Proposed Framework**.

I have expanded the mathematical depth beyond the original paper to meet thesis standards, explicitly incorporating the "Cold Start" and "Semantic Filtering" justifications derived from your Deep Research.

---

### **3.2 The Proposed Framework: Hybrid Warm-Start FixMatch**

To bridge the generalization gap formalized in Section 3.1, we propose a unified framework tailored for the computational and data constraints of agricultural edge robotics. The system, **Hybrid Warm-Start FixMatch**, is designed to operate under two critical conditions: (1) **Data Scarcity**, simulating a rapid deployment scenario where only a small budget ($B \approx 50$) of target labels can be acquired; and (2) **Latency Constraints**, requiring inference speeds compatible with real-time robotic actuation ($< 100$ ms).

The framework consists of two coupled components: a **Hybrid Active Learning Sampler** to stabilize early adaptation, and a **FixMatch-based Semi-Supervised Adapter** to filter source-specific outliers (the "Phantom Class" problem).

#### **3.2.1 Component 1: Hybrid Warm-Start Active Learning**

Standard Active Learning (AL) strategies, particularly Uncertainty Sampling (e.g., Shannon Entropy), are notoriously unstable in the early stages of domain adaptation. [cite_start]This phenomenon, known as the "Cold Start" problem[cite: 260], occurs because the source-trained model is uncalibrated on the target domain.

Let $f_{\theta_s}$ be the model pre-trained on the source domain $\mathcal{D}_s$. When applied to the target domain $\mathcal{D}_t$, the model encounters a distribution shift $P(X_s) \neq P(X_t)$. In the initial rounds ($r=0$), the model's predictive uncertainty is not a reliable proxy for "informativeness." Instead, high entropy often correlates with "outliers"—background noise or irrelevant features (e.g., soil textures) that do not exist in the source manifold. Querying these outliers wastes the annotation budget on samples that do not help define the decision boundary.

To mitigate this, we introduce a **Hybrid Warm-Start Strategy** that dynamically balances exploration (distribution matching) and exploitation (boundary refinement).

**Formulation:**
Let $\mathcal{U}_t$ be the pool of unlabeled target samples. We select a batch of $b$ samples for annotation at round $r$. The selection strategy $\mathcal{S}(x)$ is defined as a time-decaying mixture of Random Sampling ($\mathcal{S}_{rand}$) and Entropy Sampling ($\mathcal{S}_{ent}$):

$$
\mathcal{S}_{hybrid}(x, r) = \lambda_r \cdot \mathcal{S}_{rand}(x) + (1 - \lambda_r) \cdot \mathcal{S}_{ent}(x)
$$

where the decay factor $\lambda_r$ is defined as:

$$
\lambda_r = \begin{cases} 
0.3 & \text{if } r = 0 \quad \text{(Warm-Start Phase)} \\
0 & \text{if } r > 0 \quad \text{(Exploitation Phase)}
\end{cases}
$$

**The Warm-Start Phase ($r=0$):**
By forcing 30% of the initial batch to be drawn uniformly at random, we ensure that the initial labeled set $\mathcal{L}_0$ approximates the true marginal distribution of the target domain $P(Y_t)$, rather than the biased distribution implied by the source model. This "anchors" the adaptation trajectory, preventing the model from collapsing into a local minimum where it only queries background noise.

**The Exploitation Phase ($r > 0$):**
Once the cold start is broken and the model has seen a representative sample of the target classes, we switch to pure Uncertainty Sampling to refine the decision boundaries. We utilize **Shannon Entropy** as the acquisition function:

$$
x_{query} = \underset{x \in \mathcal{U}_t}{\text{argmax}} \left( - \sum_{c \in \mathcal{Y}_s} p(y=c|x; \theta) \log p(y=c|x; \theta) \right)
$$

This hybrid approach ensures that the limited budget of 50 samples is used to first *discover* the target manifold and then *refine* it, solving the failure modes observed in pure Entropy baselines.

#### **3.2.2 Component 2: FixMatch as a "Semantic Filter"**

While Active Learning adapts the model to the target domain $\mathcal{D}_t$, it does not explicitly address the **Partial Domain Adaptation (PDA)** problem—specifically, the presence of source-specific outlier classes $\mathcal{Y}_{out}$ (e.g., "Healthy" leaves in the Potato scenario) that are physically absent in the target.

Standard Domain Adaptation minimizes the global distribution discrepancy (e.g., via Maximum Mean Discrepancy or Adversarial Alignment). In a PDA setting, this is harmful: it forces the target data (which lacks $\mathcal{Y}_{out}$) to align with the full source distribution (which includes $\mathcal{Y}_{out}$). [cite_start]This mathematical forcing results in **Negative Transfer**[cite: 258], where the model "hallucinates" outlier labels on ambiguous target samples to satisfy the alignment constraint.

[cite_start]To solve this without heavy adversarial weighting mechanisms [cite: 8][cite_start], we repurpose **FixMatch** [cite: 261] as a lightweight "Semantic Filter."

**The Mechanism:**
FixMatch utilizes Consistency Regularization between a weakly augmented view $\alpha(x)$ (e.g., flip, shift) and a strongly augmented view $\mathcal{A}(x)$ (e.g., RandAugment) of the same unlabeled image. The core innovation for PDA lies in the **Confidence Thresholding**.

The unsupervised loss $\mathcal{L}_u$ is computed only on samples where the model's confidence on the weak view exceeds a strict threshold $\tau$:

$$
\mathcal{L}_u = \frac{1}{\mu B} \sum_{b=1}^{\mu B} \mathbb{1}(\max(q_b) \ge \tau) \cdot H(\hat{y}_b, p_m(y | \mathcal{A}(u_b)))
$$

where:
* $q_b = p_m(y | \alpha(u_b))$ is the prediction distribution on the weak view.
* $\hat{y}_b = \text{argmax}(q_b)$ is the pseudo-label.
* $\tau$ is the confidence threshold, set to **0.95**.
* $H$ is the cross-entropy loss.

**Why this Solves Negative Transfer:**
In a PDA scenario, target samples that visually resemble source outliers (e.g., a patch of soil that looks somewhat like a "Healthy" leaf) occupy a region of the feature space where the source model is naturally ambiguous. Unlike true disease lesions (which have distinct, high-gradient features), these "phantom" matches typically yield soft probability distributions (e.g., $p(Healthy) \approx 0.6$).

By setting a high threshold $\tau = 0.95$, these ambiguous samples fail the confidence check:

$$
\max(p(y|x_{outlier})) < 0.95 \implies \mathbb{1}(\cdot) = 0
$$

Consequently, these samples are **masked out** of the loss calculation. They do not contribute to the gradient, and the model is not forced to learn them. Conversely, samples belonging to the shared classes $\mathcal{Y}_{shared}$ (e.g., distinct Blight lesions) exhibit high semantic consistency and higher confidence, passing the threshold.

This mechanism effectively acts as a **Hard Negative Miner**, implicitly pruning the $\mathcal{Y}_{out}$ manifold from the optimization objective. Unlike "Weighting" methods that require complex auxiliary networks, this thresholding comes "for free" within the semi-supervised loop, maintaining the low computational overhead required for edge deployment.