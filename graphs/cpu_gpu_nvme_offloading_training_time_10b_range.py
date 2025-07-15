import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

categories = ['OPT-13B', 'Falcon-10B', 'Falcon-11B']

Baseline1 = [179400.41, 194424.66, 205330.16]
Mine_Approach_with_FP16 = [157649.30, 167100.59, 182709.06]
Mine_Approach_with_Both = [145080.30, 150285.71, 155424.31]

data = [Baseline1, Mine_Approach_with_FP16, Mine_Approach_with_Both]
labels = ['ZeRO-Infinity', '10Cache+FP16', '10Cache+FP16+Opt']

fig, ax = plt.subplots(figsize=(4, 1.7))

bar_width = 0.15
bar_spacing = 0.009
index = np.arange(len(categories))

colors = ['#eaa221', '#89cfef', "#1E90FF"]
patterns = ["-", "/", "\\"]

for i, d in enumerate(data):
    if patterns[i] == "":
        ax.bar(index + (i * bar_width) + (i * bar_spacing), d, bar_width,
               label=labels[i], color='white', edgecolor=colors[i],
               hatch="/", zorder=3)
    else:
        ax.bar(index + (i * bar_width) + (i * bar_spacing), d, bar_width,
               label=labels[i], color=colors[i], edgecolor='black',
               hatch=patterns[i], zorder=3)

def scientific_formatter(x, pos):
    if x == 0:
        return '0'
    power = int(np.floor(np.log10(abs(x))))
    base = x / (10 ** power)
    return f'{base:.1f}×10$^{{{power}}}$'

ax.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))

ax.set_ylabel('Training Time (s)')
ax.set_xticks(index + bar_width * 1)
ax.set_xticklabels(categories)
ax.legend(frameon=True, edgecolor='0.9', loc='lower right', fontsize=9)
ax.grid(axis='y', zorder=0, color='#C0C0C0')

plt.tight_layout()
plt.savefig('cpu_gpu_nvme_offloading_training_time_10b_range.pdf')

plt.show()