import grid2op
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lightsim2grid import LightSimBackend
from grid2op.Parameters import Parameters

# Координати вузлів IEEE 14-bus (вручну для гарного вигляду)
NODE_POS = {
    0:  (0.0, 2.0),
    1:  (1.5, 3.0),
    2:  (3.0, 3.0),
    3:  (3.0, 1.5),
    4:  (1.5, 1.0),
    5:  (5.0, 3.5),
    6:  (6.0, 2.5),
    7:  (6.5, 3.5),
    8:  (6.0, 1.5),
    9:  (5.5, 0.5),
    10: (4.5, 0.0),
    11: (5.0, 1.5),
    12: (6.5, 1.0),
    13: (7.5, 2.0),
}

def draw_grid(ax, obs, env, title="Стан мережі"):
    ax.clear()
    ax.set_facecolor('#1a1a2e')
    ax.set_title(title, fontsize=11, fontweight='bold', color='white', pad=10)

    # Малюємо лінії
    for line_id in range(env.n_line):
        or_bus = env.line_or_to_subid[line_id]
        ex_bus = env.line_ex_to_subid[line_id]
        x1, y1 = NODE_POS[or_bus]
        x2, y2 = NODE_POS[ex_bus]

        if not obs.line_status[line_id]:
            # Відключена лінія
            ax.plot([x1, x2], [y1, y2], color='#555555',
                    linewidth=1, linestyle='--', alpha=0.5)
            continue

        rho = obs.rho[line_id]
        if rho < 0.7:
            color = '#4CAF50'   # зелений — норма
            lw = 1.5
        elif rho < 0.9:
            color = '#FFC107'   # жовтий — увага
            lw = 2.5
        else:
            color = '#F44336'   # червоний — небезпека
            lw = 3.5

        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=0.9)

        # Підпис завантаженості
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, f'{rho*100:.0f}%', fontsize=6,
                color='white', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='#333333', alpha=0.7))

    # Малюємо вузли
    for node_id, (x, y) in NODE_POS.items():
        # Генератор?
        is_gen = node_id in env.gen_to_subid
        is_load = node_id in env.load_to_subid

        if is_gen:
            color = '#FFD700'
            size = 120
            marker = '*'
        elif is_load:
            color = '#64B5F6'
            size = 80
            marker = 'o'
        else:
            color = '#BBBBBB'
            size = 60
            marker = 'o'

        ax.scatter(x, y, c=color, s=size, marker=marker,
                   zorder=5, edgecolors='white', linewidths=0.5)
        ax.text(x, y + 0.15, f'Bus {node_id}', fontsize=7,
                color='white', ha='center', va='bottom')

    # Легенда
    legend_elements = [
        mpatches.Patch(color='#4CAF50', label='< 70% (норма)'),
        mpatches.Patch(color='#FFC107', label='70-90% (увага)'),
        mpatches.Patch(color='#F44336', label='> 90% (небезпека)'),
        mpatches.Patch(color='#555555', label='Відключена'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#FFD700',
                   markersize=10, label='Генератор', linewidth=0),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#64B5F6',
                   markersize=8, label='Навантаження', linewidth=0),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=7,
              facecolor='#333333', labelcolor='white', framealpha=0.8)

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 4.2)
    ax.axis('off')


# ============================
# ГОЛОВНА ДЕМОНСТРАЦІЯ
# ============================
param = Parameters()
param.NO_OVERFLOW_DISCONNECTION = False
param.NB_TIMESTEP_OVERFLOW_ALLOWED = 2

env = grid2op.make("l2rpn_case14_sandbox",
                   backend=LightSimBackend(), param=param)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#0f0f23')
fig.suptitle('Топологія енергомережі IEEE 14-bus',
             fontsize=14, fontweight='bold', color='white')

# Ліворуч — нормальний стан
obs = env.reset()
draw_grid(axes[0], obs, env, "Нормальний стан мережі")

# Праворуч — після відключення лінії
action = env.action_space({"set_line_status": [(17, -1)]})  # лінія 4_5_17
obs2, _, _, _ = env.step(action)
draw_grid(axes[1], obs2, env, "Після відключення лінії 4_5_17")

plt.tight_layout()
plt.savefig('topology.png', dpi=150, bbox_inches='tight',
            facecolor='#0f0f23')
plt.show()
print("Збережено: topology.png")