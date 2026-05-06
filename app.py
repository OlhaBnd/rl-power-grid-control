# app.py — інтерактивний дашборд керування енергомережею
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import time

from env_setup import make_grid_env, make_gym_env
from stable_baselines3 import PPO

# ── Конфігурація сторінки ──────────────────────────────
st.set_page_config(
    page_title="RL Керування Енергомережею",
    page_icon="⚡",
    layout="wide"
)

NODE_POS = {
    0:(0.0,2.0), 1:(1.5,3.0), 2:(3.0,3.0), 3:(3.0,1.5),
    4:(1.5,1.0), 5:(5.0,3.5), 6:(6.0,2.5), 7:(6.5,3.5),
    8:(6.0,1.5), 9:(5.5,0.5), 10:(4.5,0.0), 11:(5.0,1.5),
    12:(6.5,1.0), 13:(7.5,2.0),
}

# ── Завантаження моделі (кешується) ───────────────────
@st.cache_resource
def load_model():
    env = make_grid_env()
    gym_env = make_gym_env(env)
    model = PPO.load("ppo_grid2op_v2", env=gym_env)
    return model

# ── Функція малювання топології ────────────────────────
def draw_topology(obs, env, title, color_title):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    for lid in range(env.n_line):
        ob = env.line_or_to_subid[lid]
        eb = env.line_ex_to_subid[lid]
        x1, y1 = NODE_POS[ob]
        x2, y2 = NODE_POS[eb]
        if not obs.line_status[lid]:
            ax.plot([x1,x2],[y1,y2], color='#555', lw=1, ls='--', alpha=0.5)
            continue
        r = obs.rho[lid]
        color = '#4CAF50' if r < 0.7 else '#FFC107' if r < 0.9 else '#F44336'
        lw    = 1.5 if r < 0.7 else 2.5 if r < 0.9 else 3.5
        ax.plot([x1,x2],[y1,y2], color=color, lw=lw)
        ax.text((x1+x2)/2, (y1+y2)/2, f'{r*100:.0f}%',
                fontsize=6, color='white', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.1', fc='#222', alpha=0.7))

    for nid, (x, y) in NODE_POS.items():
        is_gen = nid in env.gen_to_subid
        ax.scatter(x, y,
                   c='#FFD700' if is_gen else '#64B5F6',
                   s=150 if is_gen else 80,
                   marker='*' if is_gen else 'o',
                   zorder=5, edgecolors='white', lw=0.5)
        ax.text(x, y+0.18, f'Bus {nid}', fontsize=7,
                color='white', ha='center')

    active = obs.line_status & (obs.rho > 0)
    if active.any():
        mr = np.max(obs.rho[active])
        sc = '#4CAF50' if mr < 0.7 else '#FFC107' if mr < 0.9 else '#F44336'
        ax.text(0.02, 0.02, f'Макс: {mr*100:.1f}%',
                transform=ax.transAxes, fontsize=9,
                color=sc, fontweight='bold',
                bbox=dict(boxstyle='round', fc='#333', alpha=0.8))

    ax.set_title(title, color=color_title, fontsize=11, fontweight='bold')
    ax.set_xlim(-0.5, 8.5); ax.set_ylim(-0.5, 4.2)
    ax.axis('off')
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════
#  ІНТЕРФЕЙС
# ══════════════════════════════════════════════════════
st.title("⚡ RL Керування Топологією Енергомережі")
st.markdown("**IEEE 14-bus** | PPO агент vs без агента")
st.divider()

# ── Бічна панель ───────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Налаштування")

    scenario = st.selectbox(
        "Сценарій",
        options={
            0: "Сценарій 0 — стабільний",
            3: "Сценарій 3 — критичний ⚠️",
        }.keys(),
        format_func=lambda x: {
            0: "Сценарій 0 — стабільний",
            3: "Сценарій 3 — критичний ⚠️",
        }[x],
        index=1
    )

    st.divider()
    st.subheader("🔌 Відключити лінію вручну")
    line_names = [f"Лінія {i}" for i in range(20)]
    disabled_line = st.selectbox("Вибери лінію для відключення",
                                  ["(не відключати)"] + line_names)

    st.divider()
    max_steps = st.slider("Кроків симуляції", 20, 200, 60, step=10)
    run_button = st.button("▶️ Запустити симуляцію", type="primary",
                           use_container_width=True)

# ── Основний блок ──────────────────────────────────────
if run_button:
    model = load_model()

    with st.spinner("Симуляція..."):
        env_no = make_grid_env()
        env_ag = make_grid_env()
        gym_env = make_gym_env(env_ag)
        gym_env_fresh = make_gym_env(make_grid_env())
        model_fresh = PPO.load("ppo_grid2op_v2", env=gym_env_fresh)

        # Визначаємо індекс відключеної лінії
        manual_off = None
        if disabled_line != "(не відключати)":
            manual_off = int(disabled_line.split(" ")[1])

        # Симуляція БЕЗ агента
        steps_no, rewards_no, rho_history_no = [], [], []
        obs_no = env_no.reset(options={"time serie id": scenario})

        if manual_off is not None:
            action = env_no.action_space({"set_line_status": [(manual_off, -1)]})
            obs_no, _, done_no, _ = env_no.step(action)
        else:
            done_no = False

        for _ in range(max_steps):
            if done_no: break
            obs_no, r, done_no, _ = env_no.step(env_no.action_space({}))
            steps_no.append(len(steps_no))
            rewards_no.append(r)
            active = obs_no.rho[obs_no.line_status]
            rho_history_no.append(np.max(active) if len(active) > 0 else 0.0)

        # Симуляція З агентом
        steps_ag, rewards_ag, rho_history_ag = [], [], []
        obs_gym, _ = gym_env.reset(options={"time serie id": scenario})
        obs_ag = env_ag.current_obs
        done_ag = False

        if manual_off is not None:
            attack = env_ag.action_space({"set_line_status": [(manual_off, -1)]})
            obs_ag, _, done_ag, _ = env_ag.step(attack)

        for _ in range(max_steps):
            if done_ag: break
            act, _ = model.predict(obs_gym, deterministic=True)
            obs_gym, r, term, trunc, _ = gym_env.step(act)
            obs_ag = env_ag.current_obs
            done_ag = term or trunc
            steps_ag.append(len(steps_ag))
            rewards_ag.append(r)
            active = obs_ag.rho[obs_ag.line_status]
            rho_history_ag.append(np.max(active) if len(active) > 0 else 0.0)

    # ── Метрики ────────────────────────────────────────
    st.subheader("📊 Результати симуляції")
    col1, col2, col3 = st.columns(3)

    survived_no = len(steps_no)
    survived_ag = len(steps_ag)
    ratio = survived_ag / max(survived_no, 1)

    col1.metric("Без агента", f"{survived_no} кроків",
                delta=None)
    col2.metric("PPO агент", f"{survived_ag} кроків",
                delta=f"+{survived_ag - survived_no} кроків")
    col3.metric("Покращення", f"{ratio:.1f}x",
                delta="агент кращий" if ratio > 1 else "однаково")

    st.divider()

    # ── Графік завантаженості ──────────────────────────
    st.subheader("📈 Максимальна завантаженість ліній")
    fig_rho, ax_rho = plt.subplots(figsize=(10, 3))
    fig_rho.patch.set_facecolor('#1a1a2e')
    ax_rho.set_facecolor('#1a1a2e')

    if rho_history_no:
        ax_rho.plot(rho_history_no, color='#EF5350', lw=2, label='Без агента')
    if rho_history_ag:
        ax_rho.plot(rho_history_ag, color='#42A5F5', lw=2, label='PPO агент')

    ax_rho.axhline(y=1.0, color='#FF5722', lw=1.5, ls='--', alpha=0.7,
                   label='Межа перевантаження (100%)')
    ax_rho.axhline(y=0.9, color='#FFC107', lw=1, ls=':', alpha=0.5)
    ax_rho.set_xlabel('Крок', color='white')
    ax_rho.set_ylabel('Завантаженість', color='white')
    ax_rho.tick_params(colors='white')
    ax_rho.legend(facecolor='#333', labelcolor='white')
    ax_rho.grid(True, alpha=0.2)
    for sp in ax_rho.spines.values(): sp.set_color('#444')
    plt.tight_layout()
    st.pyplot(fig_rho)

    st.divider()

    # ── Топологія ──────────────────────────────────────
    st.subheader("🗺️ Топологія мережі (фінальний стан)")
    col_no, col_ag = st.columns(2)

    with col_no:
        fig_no = draw_topology(obs_no, env_no, "Без агента", '#EF5350')
        if done_no:
            st.error("⚡ БЛЕКАУТ — мережа впала!")
        st.pyplot(fig_no)

    with col_ag:
        fig_ag = draw_topology(obs_ag, env_ag, "PPO агент", '#4CAF50')
        if done_ag:
            st.warning("⚠️ Агент не зміг утримати мережу")
        else:
            st.success(f"✅ Агент утримав мережу {survived_ag} кроків!")
        st.pyplot(fig_ag)

else:
    st.info("👈 Вибери сценарій у бічній панелі та натисни **Запустити симуляцію**")
    st.image("topology.png", caption="Приклад топології IEEE 14-bus")