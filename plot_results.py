import matplotlib.pyplot as plt
import numpy as np

# Field label budgets (0 initial + 5 rounds of 10 labels)
rounds = np.array([0, 10, 20, 30, 40, 50])

# Accuracy trajectories (%), gathered from baseline + experiments
acc_baseline = [49.0] * len(rounds)
acc_al_only = [49.0, 51.0, 53.0, 52.0, 54.0, 54.0]
acc_hybrid_fixmatch = [49.0, 56.0, 57.0, 54.0, 58.0, 62.0]

plt.figure(figsize=(8, 6))
plt.plot(rounds, acc_baseline, 'k--', label='Baseline (Source Only)', linewidth=2)
plt.plot(rounds, acc_al_only, 'b-o', label='Active Learning (Hybrid)', linewidth=2, markersize=8)
plt.plot(rounds, acc_hybrid_fixmatch, 'g-^', label='Hybrid + FixMatch (Ours)', linewidth=3, markersize=10)

plt.xlabel('Number of Field Labels', fontsize=12, fontweight='bold')
plt.ylabel('Target Domain Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Impact of Semi-Supervised Active Learning (Potato)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.ylim(45, 65)
plt.tight_layout()

output_path = 'results/figures/potato_pda_results.png'
plt.savefig(output_path, dpi=300)
print(f'Figure saved to {output_path}')

