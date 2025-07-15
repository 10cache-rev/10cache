import matplotlib.pyplot as plt
import numpy as np

categories = ['CPU', 'GPU']

opt_13 = [38, 68]
falcon_10 = [33, 74]
falcon_11 = [36, 66]

data = [opt_13, falcon_10, falcon_11]
labels = ['OPT-13B', 'Falcon-10B', 'Falcon-11B']

fig, ax = plt.subplots(figsize=(4, 1.5))

bar_width = 0.15
bar_spacing = 0.009
index = np.arange(len(categories))

colors = ['#eaa221', '#89cfef', "#1E90FF"]
patterns = ["/", "\\", "|"]

for i, d in enumerate(data):
    if patterns[i] == "":
        ax.bar(index + (i * bar_width) + (i * bar_spacing), d, bar_width, label=labels[i], color=colors[i], edgecolor=colors[i], hatch="/", zorder=3)
    else:
        ax.bar(index + (i * bar_width) + (i * bar_spacing), d, bar_width, label=labels[i], color=colors[i], edgecolor='black', hatch=patterns[i], zorder=3)

ax.set_ylabel('Utilization (%)')
ax.set_xticks(index + bar_width * 1)
ax.set_xticklabels(categories)
ax.legend(frameon=True, edgecolor='0.9')
ax.grid(axis = 'y', zorder=0, color='#C0C0C0')

plt.tight_layout()
plt.savefig('cpu_gpu_memory_uti_zero_infinity.pdf')

plt.show()