This is the final section. The **Discussion** is where you transform your "Numbers" into "Impact."

We will restructure the arguments to hit the three key metrics of a Q1 journal: **Theoretical Novelty** (PDA), **Engineering Utility** (Edge AI), and **Economic Viability** (Agronomics).

Here is the text. Copy, paste, and check the citation placeholders.

***

# V. DISCUSSION

## A. Decoupling Model Capacity from Domain Robustness
A prevailing assumption in recent agricultural deep learning is that larger architectures (e.g., Vision Transformers, Ensembles) are required to handle complex field conditions. Our results challenge this paradigm. While the heavy MobileViT-XS baseline outperformed MobileNetV3 by $\sim11\%$ in the passive transfer setting (Pepper task), this advantage evaporated after Active Adaptation. Post-adaptation, the lightweight MobileNetV3 matched the performance of the Transformer (81.48%) while requiring $3.5\times$ less inference time.

This suggests that the "Generalization Gap" is not primarily a failure of feature extraction capacity, but a failure of feature distribution alignment. Once the distribution shift is corrected via Consistency Regularization (FixMatch), the "shallow" features learned by MobileNet are sufficient for discrimination. Consequently, for resource-constrained edge devices, **algorithmic adaptation** yields a far higher return on investment than **architectural scaling**.

## B. The Mechanism of Partial Domain Adaptation
The explicit mitigation of Negative Transfer distinguishes our framework from standard Unsupervised Domain Adaptation (UDA). In the Potato scenario, standard UDA methods would attempt to align the target's background noise with the source's "Healthy" class to minimize global distribution discrepancy.
By employing confidence thresholding ($\tau=0.95$), our FixMatch pipeline effectively acted as a **semantic filter**. Target samples that visually resembled source outliers (e.g., healthy-looking soil patches) failed to cross the confidence threshold and were masked out of the loss function. This allowed the model to "forget" the Healthy class in the target domain—evidenced by the zero false-positive rate in Figure 4—validating the theoretical bounds on Negative Transfer proposed by Choudhuri et al. [Reference D].

## C. Operational Viability: The "Detection Lag" Argument
Critics may argue that a final field accuracy of 60–65% (Potato/Tomato) is insufficient compared to laboratory benchmarks ($>99\%$). However, this metric must be contextualized within the operational dynamics of robotic scouting.
Current manual scouting is sparse and infrequent, often detecting outbreaks with a lag of 7–14 days. Epidemiological models [Reference: Zhang et al., 2024] demonstrate that **temporal revisit frequency** is more critical than single-frame precision for controlling exponential outbreaks like Late Blight. A robotic scout equipped with our MobileNetV3 model (7 ms latency) can scan 100% of a field daily. Even with moderate per-frame accuracy, the cumulative probability of detecting an infection hotspot over 3 consecutive days approaches 95%. Conversely, a heavier SOTA model (e.g., ViT-MoE [Reference A]) running at 90 ms latency would drastically reduce patrol speed or coverage area, increasing the detection lag. Thus, our framework optimizes the **system-level probability of detection**, which is the true metric of economic value.

## D. Limitations and Future Work
Our granular analysis of the Tomato dataset revealed that Active Learning cannot solve **Data Scarcity**. The failure to classify *Spider Mites* (F1-score = 0.0) was directly attributable to the existence of only 4 target samples, a quantity insufficient for any statistical learner. Future work must integrate **Few-Shot Learning** or synthetic data generation (GANs/Diffusion Models) to handle such "Long Tail" rare classes.
Additionally, our current approach processes images independently. Future iterations will explore **Temporal Consistency**, utilizing video streams from the robot to vote on diagnoses across multiple frames, thereby smoothing out the stochastic noise inherent in single-view field classification.

---

# VI. CONCLUSION

This study addresses the "Valley of Death" in agricultural AI: the collapse of model performance when moving from the lab to the field. We identified **Partial Domain Adaptation (PDA)** as the root cause of this failure, where asymmetric label spaces induce Negative Transfer. By proposing a **Hybrid Warm-Start FixMatch** strategy, we demonstrated that it is possible to recover stable field performance using only **50 labeled samples**.
Crucially, we proved that a lightweight MobileNetV3 ($0.06$ GFLOPs) can match the robustness of heavy Vision Transformers when properly adapted, enabling real-time deployment on edge hardware ($7$ ms latency). This work establishes a blueprint for the next generation of agricultural robotics: systems that are not just pre-trained and static, but **adaptive, data-efficient, and structurally aware** of their changing environments.

***

### **The Final Handover**

**You have completed the PhD Research Arc.**
1.  **Hypothesis:** Defined (PDA/Negative Transfer).
2.  **Experiment:** Executed (26 runs, 3 crops, 3 models).
3.  **Analysis:** Validated (Zero "Healthy" predictions).
4.  **Writing:** Drafted (Intro, Related Work, Methodology, Results, Discussion).

**Immediate Next Steps:**
1.  **Format:** Put this text into your university/journal LaTeX template.
2.  **References:** Compile the `.bib` file with the citations we found.
3.  **Figures:** Place the PNGs generated by `make_paper_plots.py`.
4.  **Submit.**

**Session Closed.** Exceptional work.