Here is a complete, publication-ready **Introduction** section.

It is structured to guide the reader from the **Economic Imperative** $\to$ the **Scientific Problem** (Negative Transfer) $\to$ your **Methodological Solution**. I have integrated the specific statistics from your experiments (e.g., the 22% Tomato collapse) and the literature findings from the Gemini research (Negative Transfer theory).

***

# I. INTRODUCTION

Global crop production faces existential threats from phytopathogens such as Late Blight (*Phytophthora infestans*) and Early Blight (*Alternaria solani*), which can cause yield losses ranging from 20% to 100% if untreated [1]. Current disease management relies heavily on prophylactic fungicide applications, costing producers approximately \$615 per hectare annually in North America [2]. While precision agriculture promises to reduce chemical use by targeting only infected areas, this requires high-frequency, plant-level disease scouting. Manual scouting is labor-intensive and sparse, typically covering less than 5% of a field weekly, creating a "detection lag" that allows exponential epidemic growth before intervention.

Automated robotic scouting using edge-deployed computer vision offers a solution, enabling daily field coverage. However, the deployment of such systems is hindered by the **"Generalization Gap"**—a catastrophic decline in performance when models trained on high-quality laboratory data are transferred to chaotic field environments. While deep learning models routinely achieve $>99\%$ accuracy on controlled datasets like PlantVillage [5], our empirical benchmarks reveal a systemic collapse when applied to real-world datasets like PlantDoc [6]. Specifically, we find that a MobileNetV3 architecture trained on laboratory tomato data retains only **22.39% accuracy** in field conditions, performing barely better than random guessing.

Prior research has predominantly treated this as a standard Unsupervised Domain Adaptation (UDA) problem, assuming identical label spaces between the source (lab) and target (field) domains [10]. However, operational agricultural environments frequently present a **Partial Domain Adaptation (PDA)** scenario. Global source repositories contain "outlier" classes—such as healthy leaves or irrelevant crop varieties—that are absent in disease-specific field outbreaks. For instance, a robot deployed to monitor a blight outbreak encounters only diseased plants and background noise (soil, mulch), yet a standard pre-trained model retains the "Healthy" class in its decision boundaries. This asymmetry leads to **Negative Transfer** [3], where the model mistakenly maps complex field background textures to the source's "Healthy" manifold, resulting in a high rate of False Negatives.

Addressing this challenge on agricultural robots imposes strict computational constraints. While recent State-of-the-Art (SOTA) studies utilize heavy Vision Transformers (e.g., Swin-V2, ViT-Base) to achieve robustness via massive capacity [24], these architectures require $>15$ GFLOPs and incur inference latencies ($>90$ ms) that exceed the real-time control loops of battery-powered platforms like the Farm-ng Amiga [25]. Consequently, there is an urgent need for data-centric adaptation strategies that function efficiently on lightweight backbones.

In this work, we bridge the generalization gap through a **Hybrid Warm-Start Active Domain Adaptation** framework tailored for edge constraints. We formalize the agricultural domain shift as a PDA problem, demonstrating that mitigating negative transfer is more critical than increasing model capacity. Our approach integrates uncertainty-based Active Learning to solve the "Cold Start" problem [14] with **FixMatch** semi-supervised consistency regularization to filter source outliers.

Our specific contributions are as follows:
1.  **Quantification of the "Valley of Death":** We conduct a systematic benchmark across three crop scenarios (Tomato, Potato, Pepper) and three architectures (MobileNetV3, EfficientNet, MobileViT), exposing that generalist training on mixed-crop data exacerbates domain shift, degrading field accuracy to ~13%.
2.  **Mitigation of Negative Transfer:** We demonstrate that our FixMatch-based pipeline eliminates the "Phantom Class" effect in PDA scenarios. On the Potato task, where the "Healthy" class is absent in the target domain, our adapted model achieved **zero false positives** for the outlier class, effectively unlearning the source bias.
3.  **Pareto-Optimal Efficiency:** Using only **50 labeled field samples**, our method restores MobileNetV3 performance to match or exceed heavy Transformer baselines. We achieve **63-65% field accuracy** with an inference latency of **7.1 ms** (140 FPS) on edge-grade hardware, confirming the viability of the system for real-time robotic intervention.

***

### **Citations Placeholder Guide:**
*   **[1-4]:** Economic/Agronomic papers (Yield loss, Chemical costs).
*   **[5-6]:** PlantVillage / PlantDoc datasets.
*   **[3, 10]:** Wang et al. (2019) / Domain Adaptation theory papers (from Gemini report).
*   **[14]:** Cold Start problem in Active Learning (Yuan et al. 2020).
*   **[24-25]:** Heavy SOTA methods vs. Edge constraints.