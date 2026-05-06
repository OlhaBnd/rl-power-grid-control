import grid2op
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from lightsim2grid import LightSimBackend
from grid2op.Parameters import Parameters
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, DiscreteActSpace
from stable_baselines3 import PPO

NODE_POS = {
    0:  (0.0, 2.0), 1:  (1.5, 3.0), 2:  (3.0, 3.0),
    3:  (3.0, 1.5), 4:  (1.5, 1.0), 5:  (5.0, 3.5),
    6:  (6.0, 2.5), 7:  (6.5, 3.5), 8:  (6.0, 1.5),
    9:  (5.5, 0.5), 10: (4.5, 0.0), 11: (5.0, 1.5),
    12: (6.5, 1.0), 13: (7.5, 2.0),
}

def make_env():
    param = Parameters()
    param.NO_OVERFLOW_DISCONNECTION = False
    param.NB_TIMESTEP_OVERFLOW_ALLOWED = 2
    return grid2op.make("l2rpn_case14_sandbox",
                        backend=LightSimBackend(), param=param)

# Два незалежні env
env_no = make_env()
env_ag = make_env()

# gym тільки для агента
gym_env = GymEnv(env_ag)
gym_env.observation_space = BoxGymObsSpace(
    env_ag.observation_space,
    attr_to_keep=["rho", "p_or", "gen_p", "load_p", "topo_vect"]
)
gym_env.action_space = DiscreteActSpace(
    env_ag.action_space,
    attr_to_keep=["set_line_status"]
)
model = PPO.load("ppo_grid2op_v2", env=gym_env)

MAX_STEPS = 60

def collect_frames():
    frames_no, frames_ag = [], []

    # Скидаємо обидва з однаковим seed
    obs_no = env_no.reset(options={"time serie id": 3})
    obs_gym, _ = gym_env.reset(options={"time serie id": 3})
    obs_ag = env_ag.current_obs

    done_no = False
    done_ag = False

    for step in range(MAX_STEPS):
        frames_no.append({
            'rho': obs_no.rho.copy(),
            'line_status': obs_no.line_status.copy(),
            'step': step, 'done': done_no, 'label': 'БЕЗ АГЕНТА'
        })
        frames_ag.append({
            'rho': obs_ag.rho.copy(),
            'line_status': obs_ag.line_status.copy(),
            'step': step, 'done': done_ag, 'label': 'PPO АГЕНТ'
        })

        # Без агента — нічого не робить
        if not done_no:
            obs_no, _, done_no, _ = env_no.step(env_no.action_space({}))
            if done_no:
                print(f"Без агента: впала на кроці {step+1}")

        # З агентом
        if not done_ag:
            action_gym, _ = model.predict(obs_gym, deterministic=True)
            obs_gym, _, term, trunc, _ = gym_env.step(action_gym)
            obs_ag = env_ag.current_obs
            done_ag = term or trunc
            if done_ag:
                print(f"З агентом: впала на кроці {step+1}")

    if not done_ag:
        print(f"З агентом: вижила всі {MAX_STEPS} кроків!")
    if not done_no:
        print(f"Без агента: вижила всі {MAX_STEPS} кроків!")

    return frames_no, frames_ag

print("Збираємо кадри...")
frames_no, frames_ag = collect_frames()

def draw_panel(ax, frame_data):
    ax.clear()
    ax.set_facecolor('#1a1a2e')
    rho = frame_data['rho']
    line_status = frame_data['line_status']
    step = frame_data['step']
    label = frame_data['label']
    is_done = frame_data['done']

    color_label = '#F44336' if 'БЕЗ' in label else '#4CAF50'
    ax.set_title(f'{label} — крок {step}', fontsize=11,
                 fontweight='bold', color=color_label)

    if is_done and 'БЕЗ' in label:
        ax.text(0.5, 0.5, '⚡ БЛЕКАУТ ⚡', transform=ax.transAxes,
                fontsize=28, color='#F44336', fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.9))

    for line_id in range(env_no.n_line):
        or_bus = env_no.line_or_to_subid[line_id]
        ex_bus = env_no.line_ex_to_subid[line_id]
        x1, y1 = NODE_POS[or_bus]
        x2, y2 = NODE_POS[ex_bus]
        if not line_status[line_id]:
            ax.plot([x1, x2], [y1, y2], color='#555555',
                    linewidth=1, linestyle='--', alpha=0.4)
            continue
        r = rho[line_id]
        if r < 0.7:
            color, lw = '#4CAF50', 1.5
        elif r < 0.9:
            color, lw = '#FFC107', 2.5
        else:
            color, lw = '#F44336', 3.5
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=0.9)
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, f'{r*100:.0f}%', fontsize=6, color='white',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='#222222', alpha=0.7))

    for node_id, (x, y) in NODE_POS.items():
        is_gen = node_id in env_no.gen_to_subid
        color = '#FFD700' if is_gen else '#64B5F6'
        size = 150 if is_gen else 80
        marker = '*' if is_gen else 'o'
        ax.scatter(x, y, c=color, s=size, marker=marker,
                   zorder=5, edgecolors='white', linewidths=0.5)
        ax.text(x, y+0.15, f'{node_id}', fontsize=7,
                color='white', ha='center', va='bottom')

    active = line_status & (rho > 0)
    if active.any():
        max_rho = np.max(rho[active])
        sc = '#4CAF50' if max_rho < 0.7 else '#FFC107' if max_rho < 0.9 else '#F44336'
        ax.text(0.02, 0.02, f'Макс: {max_rho*100:.1f}%',
                transform=ax.transAxes, fontsize=9, color=sc, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 4.2)
    ax.axis('off')

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#0f0f23')
fig.suptitle('Порівняння: без агента vs PPO агент — однаковий сценарій',
             fontsize=12, fontweight='bold', color='white')

def animate(i):
    draw_panel(axes[0], frames_no[i])
    draw_panel(axes[1], frames_ag[i])
    return axes

print("Створюємо анімацію...")
ani = animation.FuncAnimation(fig, animate, frames=len(frames_no),
                               interval=300, blit=False, repeat=True)

ani.save('comparison_animation.gif', writer='pillow', fps=4,
         savefig_kwargs={'facecolor': '#0f0f23'})
print("Збережено: comparison_animation.gif")
plt.show()