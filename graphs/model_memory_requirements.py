import matplotlib.pyplot as plt
import numpy as np

groups = ['OPT-125M', 'OPT-350M', 'OPT-1.3B']
y0 = np.array([0.23, 0.37, 0.48])
y1 = np.array([0.25, 0.7, 2.6])
y2 = np.array([0.25, 0.7, 2.6])
y3 = np.array([0.5, 1.4, 5.2])
y4 = np.array([0.5, 1.4, 5.2])
y5 = np.array([0.5, 1.4, 5.2])
y6 = np.array([0.5, 1.4, 5.2])

fig, ax = plt.subplots(figsize=(5.5,2.15) )

ax.bar(groups, y0, label="Activation & Temp. Buffer", color="#eaa221", hatch='--', width=0.3)
ax.bar(groups, y1, bottom=y0, label="FP16 Param", color="#6693f5", hatch='//', width=0.3)
ax.bar(groups, y2, bottom=y0 + y1, label="FP16 Param Grad", color="#89cfef", hatch='\\', width=0.3)
ax.bar(groups, y3, bottom=y0 + y1 + y2, label="FP32 Param", color="#1E90FF", hatch='||', width=0.3)
ax.bar(groups, y5, bottom=y0 + y1 + y2 + y4, label="FP32 Momentum", color="#FEE8D6", hatch='X', width=0.3)
ax.bar(groups, y6, bottom=y0 + y1 + y2 + y4 + y5, label="FP32 Variance", color="#95b9c7", hatch='++', width=0.3)

ax.legend()
ax.set_ylabel('Memory (GB)')

plt.tight_layout()
plt.savefig('model_memory_requirements.pdf')

plt.show()