import matplotlib.pyplot as plt
import numpy as np

block_sizes = [256, 512, 1024]
column_names = ['Host', 'Naive', 'Shared', 'Excessive Memcpy']
data = [
    [0.154772, 0.157449, 0.15342],
    [0.00321, 0.00329, 0.003284],
    [0.003419, 0.00346, 0.003376],
    [0.05554, 0.068581, 0.053585]
]

colors = ['skyblue', 'salmon', 'lightgreen', 'lightcoral']

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

for i, ax in enumerate(axs.flat):
    ax.bar(np.arange(len(block_sizes)), data[i], color=colors)
    ax.set_xticks(np.arange(len(block_sizes)))
    ax.set_xticklabels(block_sizes)
    ax.set_title(column_names[i])
    ax.set_xlabel('Block Size')
    ax.set_ylabel('Time (ms)')

plt.tight_layout()
plt.show()
