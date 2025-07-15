import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

categories = ['L40S', 'A100']

Megatron = [None, None]
Zero_Infinity = [3903.19, 3835.028]
Stronghold = [3663.20, 5662.35]
Smart_Cache_Pref_Mem = [1499.47, 1357.49]

data = [Megatron, Zero_Infinity, Stronghold, Smart_Cache_Pref_Mem]
labels = ['Megatron', 'ZeRO-Infinity', 'StrongHold', '10Cache+P+M']

fig, ax = plt.subplots(figsize=(4, 1.6))

bar_width = 0.15
bar_spacing = 0.009
index = np.arange(len(categories))

colors = ['black', '#eaa221', '#95b9c7', "#1E90FF"]
patterns = ["", "-", '/', "\\"]

for i, d in enumerate(data):
    if i == 0:  
        for j in range(len(categories)):
            ax.text(index[j] + (i * bar_width) + (i * bar_spacing) + 0.04, 0, 'X',
                   ha='center', va='bottom', fontsize=12, color='black',
                   fontweight='bold')
    else:
        ax.bar(index + (i * bar_width) + (i * bar_spacing), d, bar_width,
               label=labels[i], color=colors[i], edgecolor='black',
               hatch=patterns[i], zorder=3)

ax.text(index[0]+ 0.18, max(Zero_Infinity) * 0.4, 'GPU OOM',
        ha='center', va='bottom', fontsize=8, fontweight='bold', color='red')

ax.set_ylabel('Training Time (s)')
ax.set_xticks(index + bar_width * 2.5)
ax.set_xticklabels(categories)
from matplotlib.patches import Patch

legend_elements = [
    Line2D([0], [0], marker='X', color='w', markerfacecolor='black', markersize=10, label='Megatron'),
    Patch(facecolor='#ffa300', edgecolor='black', hatch='-', label='ZeRO-Infinity'),
    Patch(facecolor='#95b9c7', edgecolor='black', hatch='/', label='StrongHold'),
    Patch(facecolor='#0d88e6', edgecolor='black', hatch='\\', label='10Cache+P+M')
]

ax.legend(handles=legend_elements, frameon=True, edgecolor='0.9', loc='upper center', fontsize=8)
ax.grid(axis='y', zorder=0, color='#C0C0C0')

plt.tight_layout()
plt.savefig('cpu_gpu_offload_training_time_gpt_5_9_b.pdf')

plt.show()