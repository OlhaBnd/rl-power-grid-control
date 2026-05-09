import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================
# 1. ГРАФІК НАВЧАННЯ (v2 дані)
# ============================

timesteps_v2 = [
    2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480,
    22528, 24576, 26624, 28672, 30720, 32768, 34816, 36864, 38912, 40960,
    43008, 45056, 47104, 49152, 51200, 53248, 55296, 57344, 59392, 61440,
    63488, 65536, 67584, 69632, 71680, 73728, 75776, 77824, 79872, 81920,
    83968, 86016, 88064, 90112, 92160, 94208, 96256, 98304, 100352,
    104448, 106496, 108544, 110592, 112640, 114688, 116736, 118784, 120832,
    122880, 124928, 126976, 129024, 131072, 133120, 135168, 137216, 139264,
    141312, 143360, 145408, 147456, 149504, 151552, 153600, 155648, 157696,
    159744, 161792, 163840, 165888, 167936, 169984, 172032, 174080, 176128,
    178176, 180224, 182272, 184320, 186368, 188416, 190464, 192512, 194560,
    196608, 198656, 200704
]

ep_rew_v2 = [
    180, 212, 250, 239, 303, 274, 280, 339, 434, 562,
    532, 741, 1060, 1080, 1640, 1860, 2220, 2800, 3470, 3920,
    4390, 4990, 5560, 6350, 7110, 7530, 7890, 8280, 8390, 8860,
    9690, 10100, 9920, 10100, 10400, 10900, 11400, 11500, 12000, 12400,
    12100, 12600, 12700, 12800, 12800, 13000, 13500, 13200, 13300,
    18000, 18100, 19000, 19300, 19800, 19800, 20000, 20500, 20900,
    21700, 22400, 22400, 21200, 21500, 22200, 22200, 22900, 23100,
    22500, 21400, 21600, 21600, 21100, 20900, 20500, 20400, 21000,
    20400, 19400, 18200, 17500, 17500, 18000, 18100, 18800, 19100,
    19200, 19500, 19600, 19700, 20300, 20300, 21000, 21300, 21600,
    21900, 22300, 23000
]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('PPO агент v2 — керування енергомережею IEEE 14-bus\n200 000 кроків навчання',
             fontsize=13, fontweight='bold')

# Крива навчання
ax1 = axes[0]
ax1.plot(timesteps_v2, ep_rew_v2, color='#2196F3', linewidth=1.5, alpha=0.6, label='ep_rew_mean')

window = 7
smoothed = np.convolve(ep_rew_v2, np.ones(window)/window, mode='valid')
smooth_x = timesteps_v2[window-1:]
ax1.plot(smooth_x, smoothed, color='#F44336', linewidth=2.5,
         linestyle='--', label='Згладжена (MA-7)')

ax1.set_xlabel('Кроки навчання (timesteps)')
ax1.set_ylabel('Середня винагорода за епізод')
ax1.set_title('Крива навчання')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, p: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'))

ax1.axvspan(0, 20000, alpha=0.08, color='red')
ax1.axvspan(20000, 100000, alpha=0.08, color='yellow')
ax1.axvspan(100000, 200704, alpha=0.08, color='green')
ax1.text(5000, 1000, 'Дослідження', fontsize=8, color='red', alpha=0.8)
ax1.text(45000, 1000, 'Навчання', fontsize=8, color='olive', alpha=0.8)
ax1.text(140000, 1000, 'Збіжність', fontsize=8, color='green', alpha=0.8)

# Пікова позначка
peak_idx = ep_rew_v2.index(max(ep_rew_v2))
ax1.annotate(f'Пік: {max(ep_rew_v2)/1000:.1f}k',
             xy=(timesteps_v2[peak_idx], max(ep_rew_v2)),
             xytext=(timesteps_v2[peak_idx]-30000, max(ep_rew_v2)-2000),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=9)

# ============================
# 2. ПОРІВНЯННЯ v2
# ============================
ax2 = axes[1]

episodes = ['Еп. 1', 'Еп. 2', 'Еп. 3', 'Еп. 4', 'Еп. 5', 'Середнє']
no_agent = [807, 3001, 3, 804, 513, 1026]
with_agent_v2 = [1261, 677, 806, 806, 3114, 1333]

x = np.arange(len(episodes))
width = 0.35

bars1 = ax2.bar(x - width/2, no_agent, width, label='Без агента',
                color='#EF5350', alpha=0.85)
bars2 = ax2.bar(x + width/2, with_agent_v2, width, label='PPO агент v2',
                color='#42A5F5', alpha=0.85)

for bar in bars1:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 20,
             f'{h:.0f}', ha='center', va='bottom', fontsize=7)
for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 20,
             f'{h:.0f}', ha='center', va='bottom', fontsize=7)

# Стрілка на епізод 3 — найдраматичніший випадок
ax2.annotate('3 → 806\n(268x!)',
             xy=(x[2] + width/2, 806),
             xytext=(x[2] + width/2 + 0.6, 1200),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=8, color='green', fontweight='bold')

ax2.set_xlabel('Епізод')
ax2.set_ylabel('Кроків виживання')
ax2.set_title('Порівняння: агент v2 vs без агента')
ax2.set_xticks(x)
ax2.set_xticklabels(episodes)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
ax2.get_xticklabels()[-1].set_fontweight('bold')

plt.tight_layout()
plt.savefig('learning_curve_v2.png', dpi=150, bbox_inches='tight')
plt.show()
print("Збережено: learning_curve_v2.png")