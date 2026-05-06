# demo.py — головне демо для захисту диплому
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from env_setup import make_grid_env, make_gym_env
from stable_baselines3 import PPO

print("=" * 55)
print("  Глибоке навчання з підкріпленням")
print("  Керування топологією енергомереж IEEE 14-bus")
print("=" * 55)

# ── Константи ──────────────────────────────────────────
MODEL_NAME   = "ppo_grid2op_v2"
SCENARIO_ID  = 3   # важкий сценарій
MAX_STEPS    = 80
NODE_POS = {
    0:(0.0,2.0), 1:(1.5,3.0), 2:(3.0,3.0), 3:(3.0,1.5),
    4:(1.5,1.0), 5:(5.0,3.5), 6:(6.0,2.5), 7:(6.5,3.5),
    8:(6.0,1.5), 9:(5.5,0.5), 10:(4.5,0.0),11:(5.0,1.5),
    12:(6.5,1.0),13:(7.5,2.0),
}

# ── Завантаження ────────────────────────────────────────
print("\n[1/4] Завантаження середовища та моделі...")
env_no  = make_grid_env()
env_ag  = make_grid_env()
gym_env = make_gym_env(env_ag)
model   = PPO.load(MODEL_NAME, env=gym_env)
print("      Готово ✓")

# ── Збір даних ──────────────────────────────────────────
print(f"[2/4] Симуляція сценарію {SCENARIO_ID} ({MAX_STEPS} кроків)...")

def collect(env, gym_env, model, use_agent):
    frames, done = [], False
    if use_agent:
        obs_gym, _ = gym_env.reset(options={"time serie id": SCENARIO_ID})
        obs = env.current_obs
    else:
        obs = env.reset(options={"time serie id": SCENARIO_ID})

    for step in range(MAX_STEPS):
        frames.append({
            'rho':         obs.rho.copy(),
            'line_status': obs.line_status.copy(),
            'step':        step,
            'done':        done,
        })
        if done:
            break
        if use_agent:
            act, _ = model.predict(obs_gym, deterministic=True)
            obs_gym, _, term, trunc, _ = gym_env.step(act)
            obs  = env.current_obs
            done = term or trunc
        else:
            obs, _, done, _ = env.step(env.action_space({}))

    label = "PPO агент" if use_agent else "Без агента"
    survived = frames[-1]['step']
    print(f"      {label}: {survived} кроків")
    return frames

frames_no = collect(env_no, None,    None,  False)
frames_ag = collect(env_ag, gym_env, model, True)
print("      Готово ✓")

# ── Графіки навчання ────────────────────────────────────
print("[3/4] Побудова графіків...")

timesteps = [
    2048,4096,6144,8192,10240,12288,14336,16384,18432,20480,
    22528,24576,26624,28672,30720,32768,34816,36864,38912,40960,
    43008,45056,47104,49152,51200,53248,55296,57344,59392,61440,
    63488,65536,67584,69632,71680,73728,75776,77824,79872,81920,
    83968,86016,88064,90112,92160,94208,96256,98304,100352,
    104448,106496,108544,110592,112640,114688,116736,118784,120832,
    122880,124928,126976,129024,131072,133120,135168,137216,139264,
    141312,143360,145408,147456,149504,151552,153600,155648,157696,
    159744,161792,163840,165888,167936,169984,172032,174080,176128,
    178176,180224,182272,184320,186368,188416,190464,192512,194560,
    196608,198656,200704
]
rewards = [
    180,212,250,239,303,274,280,339,434,562,532,741,1060,1080,
    1640,1860,2220,2800,3470,3920,4390,4990,5560,6350,7110,7530,
    7890,8280,8390,8860,9690,10100,9920,10100,10400,10900,11400,
    11500,12000,12400,12100,12600,12700,12800,12800,13000,13500,
    13200,13300,18000,18100,19000,19300,19800,19800,20000,20500,
    20900,21700,22400,22400,21200,21500,22200,22200,22900,23100,
    22500,21400,21600,21600,21100,20900,20500,20400,21000,20400,
    19400,18200,17500,17500,18000,18100,18800,19100,19200,19500,
    19600,19700,20300,20300,21000,21300,21600,21900,22300,23000
]

# ── Малювання ───────────────────────────────────────────
def draw_topology(ax, frames, idx, title):
    ax.clear()
    ax.set_facecolor('#1a1a2e')
    frame = frames[min(idx, len(frames)-1)]
    rho, ls = frame['rho'], frame['line_status']

    for lid in range(env_no.n_line):
        ob, eb = env_no.line_or_to_subid[lid], env_no.line_ex_to_subid[lid]
        x1,y1 = NODE_POS[ob]; x2,y2 = NODE_POS[eb]
        if not ls[lid]:
            ax.plot([x1,x2],[y1,y2],color='#555',lw=1,ls='--',alpha=0.4)
            continue
        r = rho[lid]
        color = '#4CAF50' if r<0.7 else '#FFC107' if r<0.9 else '#F44336'
        lw    = 1.5 if r<0.7 else 2.5 if r<0.9 else 3.5
        ax.plot([x1,x2],[y1,y2],color=color,lw=lw,alpha=0.9)
        ax.text((x1+x2)/2,(y1+y2)/2,f'{r*100:.0f}%',fontsize=6,
                color='white',ha='center',va='center',
                bbox=dict(boxstyle='round,pad=0.1',fc='#222',alpha=0.7))

    for nid,(x,y) in NODE_POS.items():
        is_gen = nid in env_no.gen_to_subid
        ax.scatter(x,y,c='#FFD700' if is_gen else '#64B5F6',
                   s=150 if is_gen else 80,
                   marker='*' if is_gen else 'o',
                   zorder=5,edgecolors='white',lw=0.5)
        ax.text(x,y+0.15,str(nid),fontsize=7,color='white',ha='center')

    active = ls & (rho > 0)
    if active.any():
        mr = np.max(rho[active])
        sc = '#4CAF50' if mr<0.7 else '#FFC107' if mr<0.9 else '#F44336'
        ax.text(0.02,0.02,f'Макс: {mr*100:.1f}%',transform=ax.transAxes,
                fontsize=9,color=sc,fontweight='bold',
                bbox=dict(boxstyle='round',fc='#333',alpha=0.8))

    if frame['done'] and 'Без' in title:
        ax.text(0.5,0.5,'⚡ БЛЕКАУТ ⚡',transform=ax.transAxes,
                fontsize=24,color='#F44336',fontweight='bold',
                ha='center',va='center',
                bbox=dict(boxstyle='round',fc='#1a1a2e',alpha=0.9))

    c = '#F44336' if 'Без' in title else '#4CAF50'
    ax.set_title(f'{title} — крок {frame["step"]}',
                 fontsize=10,fontweight='bold',color=c)
    ax.set_xlim(-0.5,8.5); ax.set_ylim(-0.5,4.2); ax.axis('off')

# Фігура
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('#0f0f23')
gs  = gridspec.GridSpec(2, 2, figure=fig,
                        hspace=0.35, wspace=0.25)

ax_learn = fig.add_subplot(gs[0, 0])
ax_bar   = fig.add_subplot(gs[0, 1])
ax_no    = fig.add_subplot(gs[1, 0])
ax_ag    = fig.add_subplot(gs[1, 1])

fig.suptitle('RL для керування енергомережею IEEE 14-bus  |  PPO агент',
             fontsize=14, fontweight='bold', color='white')

# Крива навчання
ax_learn.set_facecolor('#1a1a2e')
ax_learn.plot(timesteps, rewards, color='#2196F3', lw=1.5, alpha=0.6)
w = 7
sm = np.convolve(rewards, np.ones(w)/w, mode='valid')
ax_learn.plot(timesteps[w-1:], sm, color='#FF5722', lw=2.5, ls='--')
ax_learn.set_title('Крива навчання', color='white', fontsize=10)
ax_learn.set_xlabel('Кроки', color='white', fontsize=8)
ax_learn.set_ylabel('Винагорода', color='white', fontsize=8)
ax_learn.tick_params(colors='white', labelsize=7)
ax_learn.grid(True, alpha=0.2)
for sp in ax_learn.spines.values(): sp.set_color('#444')
ax_learn.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x,p: f'{x/1000:.0f}k' if x>=1000 else f'{x:.0f}'))

# Стовпчиковий графік
ax_bar.set_facecolor('#1a1a2e')
methods  = ['Без агента', 'Random', 'PPO агент']
avg_steps= [1026, 16, 1162]
colors   = ['#EF5350', '#FF9800', '#42A5F5']
bars = ax_bar.bar(methods, avg_steps, color=colors, alpha=0.85, width=0.5)
for bar, val in zip(bars, avg_steps):
    ax_bar.text(bar.get_x()+bar.get_width()/2, val+15,
                str(val), ha='center', va='bottom',
                color='white', fontsize=9, fontweight='bold')
ax_bar.set_title('Порівняння методів (сер. кроків)', color='white', fontsize=10)
ax_bar.set_ylabel('Кроків виживання', color='white', fontsize=8)
ax_bar.tick_params(colors='white', labelsize=8)
ax_bar.grid(True, alpha=0.2, axis='y')
for sp in ax_bar.spines.values(): sp.set_color('#444')

# Анімація топологій
import matplotlib.animation as animation

def animate(i):
    draw_topology(ax_no, frames_no, i, 'Без агента')
    draw_topology(ax_ag, frames_ag, i, 'PPO агент')
    return ax_no, ax_ag

ani = animation.FuncAnimation(fig, animate,
                               frames=MAX_STEPS,
                               interval=250,
                               blit=False,
                               repeat=True)

print("      Готово ✓")
print("[4/4] Запуск демо...")
print("\n      Натисни Ctrl+C або закрий вікно щоб завершити\n")
plt.show()