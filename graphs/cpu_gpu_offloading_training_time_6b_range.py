import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, FuncFormatter

categories = ['OPT-6.7B', 'Bloom-7B', 'Falcon-7B']

Baseline2 = [None, None, None]
Baseline1 = [4469.19, 9444.80, 27671.92]
Smart_Cache = [5078.11, 10789.78, 29119.76]
Smart_Cache_Pref = [3538.3, 7673.4, 24846.36]
Smart_Cache_Pref_Mem = [1758.34, 3435.88, 10581.84]
l2l = [10562.41, 18902.54, 55518.38]

data = [Baseline2, Baseline1, l2l, Smart_Cache, Smart_Cache_Pref, Smart_Cache_Pref_Mem]
labels = ['ZeRO-Offload', 'ZeRO-Infinity', 'L2L', '10Cache', '10Cache+P', '10Cache+P+M']

fig, ax = plt.subplots(figsize=(4, 2.1))

bar_width = 0.15
bar_spacing = 0.009
index = np.arange(len(categories))

colors = ['black', '#eaa221', '#FEE8D6', '#6693f5', "#89cfef", "#1E90FF"]
patterns = ["", "--", 'XX', "//", "\\\\", "++"]

for i, d in enumerate(data):
    if i == 0:  
        for j in range(len(categories)):
            ax.text(index[j] + (i * bar_width) + (i * bar_spacing), 0, 'X',
                   ha='center', va='bottom', fontsize=12, color='black',
                   fontweight='bold')
    else:
        ax.bar(index + (i * bar_width) + (i * bar_spacing), d, bar_width,
               label=labels[i], color=colors[i], edgecolor='black',
               hatch=patterns[i], zorder=3)

ax.text(index[0] + 0.27, max(Baseline1) * 0.4, 'GPU OOM',
        ha='center', va='bottom', fontsize=8, fontweight='bold', color='red')

def scientific_formatter(x, pos):
    if x == 0:
        return '0'
    power = int(np.floor(np.log10(abs(x))))
    base = x / (10 ** power)
    return f'{base:.1f}×10$^{{{power}}}$'

ax.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))

ax.set_ylabel('Training Time (s)')
ax.set_xticks(index + bar_width * 2.5)
ax.set_xticklabels(categories)

legend_elements = [
    Line2D([0], [0], marker='X', color='w', markerfacecolor='black', markersize=10, label='ZeRO-Offload'),
    Patch(facecolor=colors[1], edgecolor='black', hatch=patterns[1], label='ZeRO-Infinity'),
    Patch(facecolor=colors[2], edgecolor='black', hatch=patterns[2], label='L2L'),
    Patch(facecolor=colors[3], edgecolor='black', hatch=patterns[3], label='10Cache'),
    Patch(facecolor=colors[4], edgecolor='black', hatch=patterns[4], label='10Cache+P'),
    Patch(facecolor=colors[5], edgecolor='black', hatch=patterns[5], label='10Cache+P+M'),
]

ax.legend(handles=legend_elements, frameon=True, edgecolor='0.9', loc='upper center', fontsize=8)
ax.grid(axis='y', zorder=0, color='#C0C0C0')

plt.tight_layout()
plt.savefig('cpu_gpu_offloading_training_time_6b_range.pdf')

plt.show()