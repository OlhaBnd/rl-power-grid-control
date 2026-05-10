# app.py — інтерактивний дашборд керування енергомережею
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.env_setup import make_grid_env, make_gym_env
from stable_baselines3 import PPO
import json
import os

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

@st.cache_resource
def load_model():
    env = make_grid_env()
    gym_env = make_gym_env(env)
    model = PPO.load("models/ppo_grid2op_v3", env=gym_env)
    return model

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
        lw = 1.5 if r < 0.7 else 2.5 if r < 0.9 else 3.5
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
                transform=ax.transAxes, fontsize=9, color=sc, fontweight='bold',
                bbox=dict(boxstyle='round', fc='#333', alpha=0.8))
    ax.set_title(title, color=color_title, fontsize=11, fontweight='bold')
    ax.set_xlim(-0.5, 8.5); ax.set_ylim(-0.5, 4.2)
    ax.axis('off')
    plt.tight_layout()
    return fig

@st.cache_resource
def scan_scenarios():
    import grid2op
    from lightsim2grid import LightSimBackend
    from grid2op.Parameters import Parameters
    param = Parameters()
    param.NO_OVERFLOW_DISCONNECTION = False
    param.NB_TIMESTEP_OVERFLOW_ALLOWED = 2
    env_scan = grid2op.make("l2rpn_case14_sandbox",
                            backend=LightSimBackend(), param=param)
    results = {}
    for sid in range(20):
        obs = env_scan.reset(options={"time serie id": sid})
        done = False
        steps = 0
        while not done and steps < 200:
            obs, _, done, _ = env_scan.step(env_scan.action_space({}))
            steps += 1
        if steps < 10:
            label = f"Сценарій {sid} — критичний ⚠️"
        elif steps < 100:
            label = f"Сценарій {sid} — важкий"
        else:
            label = f"Сценарій {sid} — стабільний"
        results[sid] = label
    return results

# ══════════════════════════════════════════════════════
st.title("⚡ RL Керування Топологією Енергомережі")
st.markdown("**IEEE 14-bus** | PPO агент vs Greedy vs без агента")
st.divider()

with st.sidebar:
    st.header("⚙️ Налаштування")

    with st.spinner("Завантаження сценаріїв..."):
        scenario_labels = scan_scenarios()

    scenario = st.selectbox(
        "Сценарій",
        options=list(scenario_labels.keys()),
        format_func=lambda x: scenario_labels[x],
        index=3
    )

    st.divider()
    st.subheader("⚡ Сценарій аварії")
    
    attack_type = st.radio(
        "Тип аварії",
        ["Без аварії", "Одна лінія", "Каскадна аварія (N-1-1)"],
        index=0
    )

    line_names = [f"Лінія {i}" for i in range(20)]
    
    disabled_line = "(не відключати)"
    disabled_lines_cascade = []

    if attack_type == "Одна лінія":
        disabled_line = st.selectbox(
            "Вибери лінію для відключення",
            ["(не відключати)"] + line_names
        )
    elif attack_type == "Каскадна аварія (N-1-1)":
        st.markdown("*N-1-1: дві лінії відключаються одна за одною*")
        line1 = st.selectbox("Перша лінія (відключається одразу)", line_names, index=17)
        line2 = st.selectbox("Друга лінія (відключається через 3 кроки)", line_names, index=4)
        disabled_lines_cascade = [int(line1.split(" ")[1]), int(line2.split(" ")[1])]

    st.divider()
    max_steps = st.slider("Кроків симуляції", 20, 200, 100, step=10)
    run_button = st.button("▶️ Запустити симуляцію", type="primary",
                           use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs(["🔬 Симуляція", "📊 Порівняння сценаріїв", "🏗️ Архітектура", "🕹️ Ручне керування"])

with tab1:
    if run_button:
        model = load_model()

        with st.spinner("Симуляція..."):
            env_no  = make_grid_env()
            env_ag  = make_grid_env()
            env_grd = make_grid_env()
            gym_env = make_gym_env(env_ag)

            # ── Визначаємо тип аварії ──────────────────
            manual_off = None
            if attack_type == "Одна лінія" and disabled_line != "(не відключати)":
                manual_off = int(disabled_line.split(" ")[1])

            # ── БЕЗ АГЕНТА ────────────────────────────
            steps_no, rho_history_no = [], []
            obs_no = env_no.reset(options={"time serie id": scenario})
            done_no = False
            cascade_step_no = 0

            if attack_type == "Каскадна аварія (N-1-1)" and disabled_lines_cascade:
                obs_no, _, done_no, _ = env_no.step(
                    env_no.action_space({"set_line_status": [(disabled_lines_cascade[0], -1)]})
                )
            elif manual_off is not None:
                obs_no, _, done_no, _ = env_no.step(
                    env_no.action_space({"set_line_status": [(manual_off, -1)]})
                )

            for _ in range(max_steps):
                if done_no: break
                if (attack_type == "Каскадна аварія (N-1-1)"
                        and disabled_lines_cascade
                        and cascade_step_no == 3
                        and obs_no.line_status[disabled_lines_cascade[1]]):
                    obs_no, _, done_no, _ = env_no.step(
                        env_no.action_space({"set_line_status": [(disabled_lines_cascade[1], -1)]})
                    )
                else:
                    obs_no, _, done_no, _ = env_no.step(env_no.action_space({}))
                cascade_step_no += 1
                steps_no.append(len(steps_no))
                active = obs_no.rho[obs_no.line_status]
                rho_history_no.append(np.max(active) if len(active) > 0 else 0.0)

            # ── PPO АГЕНТ ──────────────────────────────
            steps_ag, rho_history_ag = [], []
            actions_count_ag = 0
            event_log = []
            cascade_step_ag = 0

            obs_gym, _ = gym_env.reset(options={"time serie id": scenario})
            obs_ag = env_ag.current_obs
            done_ag = False

            if attack_type == "Каскадна аварія (N-1-1)" and disabled_lines_cascade:
                obs_ag, _, done_ag, _ = env_ag.step(
                    env_ag.action_space({"set_line_status": [(disabled_lines_cascade[0], -1)]})
                )
            elif manual_off is not None:
                obs_ag, _, done_ag, _ = env_ag.step(
                    env_ag.action_space({"set_line_status": [(manual_off, -1)]})
                )

            prev_status_ag = obs_ag.line_status.copy()

            for _ in range(max_steps):
                if done_ag: break

                # Каскадне відключення другої лінії через 3 кроки
                if (attack_type == "Каскадна аварія (N-1-1)"
                        and disabled_lines_cascade
                        and cascade_step_ag == 3
                        and obs_ag.line_status[disabled_lines_cascade[1]]):
                    obs_ag, _, done_ag, _ = env_ag.step(
                        env_ag.action_space({"set_line_status": [(disabled_lines_cascade[1], -1)]})
                    )
                    # Синхронізуємо gym_env
                    obs_gym, _, term, trunc, _ = gym_env.step(
                        gym_env.action_space.sample() * 0
                    )
                    obs_ag = env_ag.current_obs
                    done_ag = term or trunc
                else:
                    act, _ = model.predict(obs_gym, deterministic=True)
                    obs_gym, _, term, trunc, _ = gym_env.step(act)
                    obs_ag = env_ag.current_obs
                    done_ag = term or trunc

                cascade_step_ag += 1

                changed = np.where(obs_ag.line_status != prev_status_ag)[0]
                for line_id in changed:
                    status = "✅ підключено" if obs_ag.line_status[line_id] else "❌ відключено"
                    active = obs_ag.rho[obs_ag.line_status]
                    max_rho = np.max(active) if len(active) > 0 else 0
                    event_log.append({
                        'крок': len(steps_ag),
                        'лінія': f"Лінія {line_id}",
                        'дія': status,
                        'макс_rho': f"{max_rho*100:.1f}%"
                    })
                    actions_count_ag += 1
                prev_status_ag = obs_ag.line_status.copy()
                steps_ag.append(len(steps_ag))
                active = obs_ag.rho[obs_ag.line_status]
                rho_history_ag.append(np.max(active) if len(active) > 0 else 0.0)

            # ── GREEDY АГЕНТ ───────────────────────────
            steps_grd, rho_history_grd = [], []
            actions_count_grd = 0
            cascade_step_grd = 0

            obs_grd = env_grd.reset(options={"time serie id": scenario})
            done_grd = False

            if attack_type == "Каскадна аварія (N-1-1)" and disabled_lines_cascade:
                obs_grd, _, done_grd, _ = env_grd.step(
                    env_grd.action_space({"set_line_status": [(disabled_lines_cascade[0], -1)]})
                )
            elif manual_off is not None:
                obs_grd, _, done_grd, _ = env_grd.step(
                    env_grd.action_space({"set_line_status": [(manual_off, -1)]})
                )

            prev_status_grd = obs_grd.line_status.copy()

            for _ in range(max_steps):
                if done_grd: break

                if (attack_type == "Каскадна аварія (N-1-1)"
                        and disabled_lines_cascade
                        and cascade_step_grd == 3
                        and obs_grd.line_status[disabled_lines_cascade[1]]):
                    obs_grd, _, done_grd, _ = env_grd.step(
                        env_grd.action_space({"set_line_status": [(disabled_lines_cascade[1], -1)]})
                    )
                else:
                    rho = obs_grd.rho.copy()
                    rho[~obs_grd.line_status] = 0
                    if np.max(rho) > 0.9:
                        line_off = int(np.argmax(rho))
                        action_grd = env_grd.action_space(
                            {"set_line_status": [(line_off, -1)]}
                        )
                    else:
                        action_grd = env_grd.action_space({})
                    obs_grd, _, done_grd, _ = env_grd.step(action_grd)

                cascade_step_grd += 1
                if not np.array_equal(obs_grd.line_status, prev_status_grd):
                    actions_count_grd += 1
                prev_status_grd = obs_grd.line_status.copy()
                steps_grd.append(len(steps_grd))
                active = obs_grd.rho[obs_grd.line_status]
                rho_history_grd.append(np.max(active) if len(active) > 0 else 0.0)

        # ── МЕТРИКИ ────────────────────────────────────
        survived_no  = len(steps_no)
        survived_grd = len(steps_grd)
        survived_ag  = len(steps_ag)

        attack_label = {
            "Без аварії": "штатний режим",
            "Одна лінія": f"відключення {disabled_line}",
            "Каскадна аварія (N-1-1)": f"каскад: Лінія {disabled_lines_cascade[0]} → Лінія {disabled_lines_cascade[1]}" if disabled_lines_cascade else ""
        }.get(attack_type, "")

        st.subheader(f"📊 Результати симуляції — {attack_label}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Без агента", f"{survived_no} кроків")
        col2.metric("Greedy агент", f"{survived_grd} кроків",
                    delta=f"{survived_grd - survived_no:+d} кроків")
        col3.metric("PPO агент", f"{survived_ag} кроків",
                    delta=f"{survived_ag - survived_no:+d} кроків")
        col4.metric("PPO vs Greedy",
                    f"{survived_ag/max(survived_grd,1):.1f}x",
                    delta="PPO кращий" if survived_ag > survived_grd else "однаково")

        st.subheader("📉 Детальні метрики")
        m1, m2, m3, m4 = st.columns(4)
        avg_rho_no  = np.mean(rho_history_no)  if rho_history_no  else 0
        avg_rho_grd = np.mean(rho_history_grd) if rho_history_grd else 0
        avg_rho_ag  = np.mean(rho_history_ag)  if rho_history_ag  else 0
        m1.metric("Сер. завантаженість (без агента)", f"{avg_rho_no*100:.1f}%")
        m2.metric("Сер. завантаженість (Greedy)",
                  f"{avg_rho_grd*100:.1f}%",
                  delta=f"{(avg_rho_grd-avg_rho_no)*100:.1f}% (нижче = краще)",
                  delta_color="inverse")
        m3.metric("Сер. завантаженість (PPO)",
                  f"{avg_rho_ag*100:.1f}%",
                  delta=f"{(avg_rho_ag-avg_rho_no)*100:.1f}% (нижче = краще)",
                  delta_color="inverse")
        m4.metric("Перемикань PPO / Greedy",
                  f"{actions_count_ag} / {actions_count_grd}")

        st.divider()

        # ── ЖУРНАЛ ПОДІЙ ───────────────────────────────
        st.subheader("📋 Журнал дій PPO агента")
        if event_log:
            df_log = pd.DataFrame(event_log)
            df_log.columns = ['Крок', 'Лінія', 'Дія', 'Макс. завантаженість']
            st.dataframe(df_log, use_container_width=True, hide_index=True)
            st.caption(
                f"Всього дій: {len(event_log)} | "
                f"Підключень: {sum(1 for e in event_log if '✅' in e['дія'])} | "
                f"Відключень: {sum(1 for e in event_log if '❌' in e['дія'])}"
            )
        else:
            st.info("Агент не виконував перемикань — мережа була стабільною.")

        st.divider()

        # ── ГРАФІК ЗАВАНТАЖЕНОСТІ ──────────────────────
        st.subheader("📈 Максимальна завантаженість ліній")
        fig_rho, ax_rho = plt.subplots(figsize=(10, 3))
        fig_rho.patch.set_facecolor('#1a1a2e')
        ax_rho.set_facecolor('#1a1a2e')
        if rho_history_no:
            ax_rho.plot(rho_history_no, color='#EF5350', lw=2, label='Без агента')
        if rho_history_ag:
            ax_rho.plot(rho_history_ag, color='#42A5F5', lw=2, label='PPO агент')
        if rho_history_grd:
            ax_rho.plot(rho_history_grd, color='#FF9800', lw=2,
                        label='Greedy агент', linestyle='-.')
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

        # ── ТОПОЛОГІЯ ──────────────────────────────────
        st.subheader("🗺️ Топологія мережі (фінальний стан)")
        col_no, col_grd, col_ag = st.columns(3)
        with col_no:
            fig_no = draw_topology(obs_no, env_no, "Без агента", '#EF5350')
            if done_no:
                st.error("⚡ БЛЕКАУТ!")
            st.pyplot(fig_no)
        with col_grd:
            fig_grd = draw_topology(obs_grd, env_grd, "Greedy агент", '#FF9800')
            if done_grd:
                st.error("⚡ БЛЕКАУТ!")
            else:
                st.info(f"🟡 Greedy: {survived_grd} кроків")
            st.pyplot(fig_grd)
        with col_ag:
            fig_ag = draw_topology(obs_ag, env_ag, "PPO агент", '#4CAF50')
            if survived_ag > survived_no:
                st.success(f"✅ PPO: {survived_ag} кроків")
            elif survived_ag == survived_no:
                st.info(f"ℹ️ PPO: {survived_ag} кроків")
            else:
                st.warning(f"⚠️ PPO: {survived_ag} кроків")
            st.pyplot(fig_ag)

        st.divider()

        # ── КАРТА РИЗИКІВ ──────────────────────────────
        st.subheader("🔥 Карта ризиків вузлів")
        node_risk = np.zeros(14)
        node_risk_count = np.zeros(14)
        for lid in range(env_ag.n_line):
            or_bus = env_ag.line_or_to_subid[lid]
            ex_bus = env_ag.line_ex_to_subid[lid]
            r = obs_ag.rho[lid] if obs_ag.line_status[lid] else 0
            node_risk[or_bus] += r
            node_risk[ex_bus] += r
            node_risk_count[or_bus] += 1
            node_risk_count[ex_bus] += 1
        node_risk_norm = np.divide(
            node_risk, node_risk_count,
            where=node_risk_count > 0
        )
        fig_risk, ax_risk = plt.subplots(figsize=(10, 6))
        fig_risk.patch.set_facecolor('#1a1a2e')
        ax_risk.set_facecolor('#1a1a2e')
        for lid in range(env_ag.n_line):
            or_bus = env_ag.line_or_to_subid[lid]
            ex_bus = env_ag.line_ex_to_subid[lid]
            x1, y1 = NODE_POS[or_bus]
            x2, y2 = NODE_POS[ex_bus]
            r = obs_ag.rho[lid] if obs_ag.line_status[lid] else 0
            color = '#4CAF50' if r < 0.7 else '#FFC107' if r < 0.9 else '#F44336'
            ax_risk.plot([x1,x2],[y1,y2], color=color, lw=1.5, alpha=0.5)
        cmap = plt.cm.RdYlGn_r
        for nid, (x, y) in NODE_POS.items():
            risk = node_risk_norm[nid]
            color = cmap(min(risk, 1.0))
            size = 200 + risk * 800
            ax_risk.scatter(x, y, c=[color], s=size,
                            zorder=5, edgecolors='white', lw=1.5, alpha=0.9)
            ax_risk.text(x, y, f"Bus {nid}\n{risk*100:.0f}%",
                        fontsize=7, color='white',
                        ha='center', va='center', fontweight='bold')
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                    norm=plt.Normalize(vmin=0, vmax=100))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_risk, shrink=0.6, pad=0.02)
        cbar.set_label('Ризик (%)', color='white', fontsize=9)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
        ax_risk.set_title('Карта ризиків вузлів',
                          color='white', fontsize=11, fontweight='bold')
        ax_risk.set_xlim(-0.8, 9.0); ax_risk.set_ylim(-0.8, 4.5)
        ax_risk.axis('off')
        plt.tight_layout()
        st.pyplot(fig_risk)

        top3 = np.argsort(node_risk_norm)[::-1][:3]
        col_r1, col_r2, col_r3 = st.columns(3)
        for i, (col, nid) in enumerate(zip([col_r1, col_r2, col_r3], top3)):
            risk_pct = node_risk_norm[nid] * 100
            is_gen = nid in env_ag.gen_to_subid
            node_type = "⚡ Генератор" if is_gen else "🔵 Навантаження"
            col.metric(f"#{i+1} Найризикованіший: Bus {nid}",
                       f"{risk_pct:.1f}% ризик",
                       delta=node_type, delta_color="off")

    else:
        st.info("👈 Вибери сценарій у бічній панелі та натисни **Запустити симуляцію**")
        st.image("outputs/topology.png", caption="Приклад топології IEEE 14-bus")

with tab2:
    st.subheader("📊 Порівняння всіх 20 сценаріїв")

    cache_file = "outputs/benchmark_results.json"

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        load_cache = st.button("⚡ Завантажити збережені результати",
                               type="primary", use_container_width=True)
    with col_btn2:
        run_fresh = st.button("🔄 Перерахувати заново (~3 хв)",
                              use_container_width=True)

    if load_cache and os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            rows = json.load(f)
        st.success("✅ Результати завантажено миттєво!")
        show_results = True

    elif run_fresh:
        with st.spinner("Тестування всіх сценаріїв (~3 хвилини)..."):
            rows = []
            progress = st.progress(0)
            for sid in range(20):
                env_base = make_grid_env()
                obs = env_base.reset(options={"time serie id": sid})
                done, steps_no = False, 0
                while not done and steps_no < 300:
                    obs, _, done, _ = env_base.step(env_base.action_space({}))
                    steps_no += 1

                env_ag  = make_grid_env()
                gym_ag  = make_gym_env(env_ag)
                model_s = PPO.load("models/ppo_grid2op_v3", env=gym_ag)
                obs_gym, _ = gym_ag.reset(options={"time serie id": sid})
                done, steps_ag = False, 0
                while not done and steps_ag < 300:
                    act, _ = model_s.predict(obs_gym, deterministic=True)
                    obs_gym, _, term, trunc, _ = gym_ag.step(act)
                    done = term or trunc
                    steps_ag += 1

                env_grd = make_grid_env()
                obs_grd = env_grd.reset(options={"time serie id": sid})
                done, steps_grd = False, 0
                while not done and steps_grd < 300:
                    rho = obs_grd.rho.copy()
                    rho[~obs_grd.line_status] = 0
                    if np.max(rho) > 0.9:
                        act_grd = env_grd.action_space(
                            {"set_line_status": [(int(np.argmax(rho)), -1)]}
                        )
                    else:
                        act_grd = env_grd.action_space({})
                    obs_grd, _, done, _ = env_grd.step(act_grd)
                    steps_grd += 1

                label = scenario_labels[sid].split(" — ")[1]
                rows.append({
                    "scenario_id": sid,
                    "type": label,
                    "no_agent": steps_no,
                    "greedy": steps_grd,
                    "ppo": steps_ag,
                    "ppo_vs_no": round(steps_ag/max(steps_no,1), 2),
                    "ppo_vs_greedy": round(steps_ag/max(steps_grd,1), 2),
                })
                progress.progress((sid+1)/20)

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            st.success("✅ Результати збережені!")
        show_results = True

    else:
        show_results = False
        if not os.path.exists(cache_file):
            st.warning("⚠️ Збережених результатів немає. Натисни 'Перерахувати заново'")
        else:
            st.info("Натисни '⚡ Завантажити збережені результати' для миттєвого завантаження")

    if show_results and rows:
        df = pd.DataFrame(rows)
        df.columns = ["Сценарій", "Тип", "Без агента",
                      "Greedy", "PPO агент",
                      "PPO vs без агента", "PPO vs Greedy"]
        st.dataframe(df, use_container_width=True, hide_index=True)

        fig_all, ax_all = plt.subplots(figsize=(12, 5))
        fig_all.patch.set_facecolor('#1a1a2e')
        ax_all.set_facecolor('#1a1a2e')
        x = np.arange(20)
        w = 0.25
        no_vals  = [r["no_agent"] if isinstance(r, dict) else r[2] for r in rows]
        grd_vals = [r["greedy"]   if isinstance(r, dict) else r[3] for r in rows]
        ppo_vals = [r["ppo"]      if isinstance(r, dict) else r[4] for r in rows]
        ax_all.bar(x-w, no_vals,  w, label='Без агента',  color='#EF5350', alpha=0.85)
        ax_all.bar(x,   grd_vals, w, label='Greedy',      color='#FF9800', alpha=0.85)
        ax_all.bar(x+w, ppo_vals, w, label='PPO агент',   color='#42A5F5', alpha=0.85)
        ax_all.set_xlabel('Сценарій', color='white')
        ax_all.set_ylabel('Кроків виживання', color='white')
        ax_all.set_title('Порівняння всіх 20 сценаріїв',
                         color='white', fontsize=12)
        ax_all.set_xticks(x)
        ax_all.set_xticklabels([str(i) for i in range(20)])
        ax_all.tick_params(colors='white')
        ax_all.legend(facecolor='#333', labelcolor='white')
        ax_all.grid(True, alpha=0.2, axis='y')
        for sp in ax_all.spines.values(): sp.set_color('#444')
        plt.tight_layout()
        st.pyplot(fig_all)

        avg_ppo = np.mean(ppo_vals)
        avg_no  = np.mean(no_vals)
        avg_grd = np.mean(grd_vals)
        c1, c2, c3 = st.columns(3)
        c1.metric("Сер. кроків без агента", f"{avg_no:.0f}")
        c2.metric("Сер. кроків Greedy",     f"{avg_grd:.0f}")
        c3.metric("Сер. кроків PPO",        f"{avg_ppo:.0f}",
                  delta=f"+{avg_ppo-avg_no:.0f} vs без агента")
        
with tab3:
    st.subheader("🏗️ Архітектура системи")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 🎯 Reward Function")
        st.latex(r"R(t) = \alpha \cdot S(t) - \beta \cdot O(t) - \gamma \cdot L(t) + \delta \cdot B(t)")
        reward_data = {
            "Компонент": ["S(t)", "O(t)", "L(t)", "B(t)"],
            "Назва": ["Стабільність", "Штраф перевантаження", "Штраф блекауту", "Бонус з'єднаності"],
            "Коефіцієнт": ["α = 1.0", "β = 2.0", "γ = 100.0", "δ = 0.5"],
            "Опис": [
                "+1 за кожен крок без аварії",
                "-2 × сума перевантажень > 90%",
                "-100 при блекауті",
                "+0.5 × частка активних ліній"
            ]
        }
        st.dataframe(reward_data, use_container_width=True, hide_index=True)

        st.markdown("### ⚙️ Гіперпараметри PPO")
        ppo_params = {
            "Параметр": ["learning_rate", "n_steps", "batch_size", "n_epochs", "ent_coef", "clip_range", "gamma"],
            "Значення": ["3×10⁻⁴", "2048", "64", "10", "0.01", "0.2", "0.99"],
            "Призначення": [
                "Швидкість навчання",
                "Кроків до оновлення",
                "Розмір міні-батчу",
                "Епох на оновлення",
                "Коефіцієнт ентропії",
                "Діапазон кліпування",
                "Дисконт майбутніх нагород"
            ]
        }
        st.dataframe(ppo_params, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("### 📈 Порівняння версій моделі")
        versions_data = {
            "Версія": ["v1 (100k кроків)", "v2 (200k кроків)", "v3 (300k + власна reward)"],
            "Reward Function": [
                "L2RPNReward (стандартна)",
                "L2RPNReward (стандартна)",
                "CustomReward (власна)"
            ],
            "Сер. кроків": [192, 1162, 2052],
            "Критичний сценарій": ["~13x", "~518x", "1661x"],
        }
        st.dataframe(versions_data, use_container_width=True, hide_index=True)
        st.success("✅ v3 з власною reward function показує найкращі результати")

        st.markdown("### 🧠 Нейромережа PPO (MlpPolicy)")
        st.markdown("""
| Компонент | Деталі |
|-----------|--------|
| Тип | Actor-Critic (2 мережі) |
| Шари | 2 приховані × 64 нейрони |
| Активація | tanh |
| Вхід | 114 параметрів |
| Вихід Actor | 101 дія (softmax) |
| Вихід Critic | 1 значення V(s) |
| Параметрів | ~30 000 |
        """)       
with tab4:
    st.subheader("🕹️ Ручне керування мережею")
    st.markdown("Керуй мережею самостійно і порівняй результат з PPO агентом")

    col_ctrl_l, col_ctrl_r = st.columns([1, 2])

    with col_ctrl_l:
        st.markdown("### ⚙️ Налаштування")

        manual_scenario = st.selectbox(
            "Сценарій",
            options=list(scenario_labels.keys()),
            format_func=lambda x: scenario_labels[x],
            index=3,
            key="manual_scenario"
        )

        manual_steps = st.slider(
            "Кроків симуляції", 10, 100, 30,
            key="manual_steps"
        )

        st.markdown("### 🔌 Вибери активні лінії")
        st.markdown("*Зніми галочку щоб відключити лінію*")

        # Ініціалізуємо стан ліній
        if "line_states" not in st.session_state:
            st.session_state.line_states = [True] * 20

        line_cols = st.columns(2)
        for i in range(20):
            col_idx = i % 2
            with line_cols[col_idx]:
                st.session_state.line_states[i] = st.checkbox(
                    f"Лінія {i}",
                    value=st.session_state.line_states[i],
                    key=f"line_{i}"
                )

        run_manual = st.button(
            "▶️ Запустити ручну симуляцію",
            type="primary",
            use_container_width=True,
            key="run_manual"
        )

    with col_ctrl_r:
        if run_manual:
            with st.spinner("Симуляція..."):
                # ── РУЧНЕ КЕРУВАННЯ ────────────────────
                env_man = make_grid_env()
                env_ppo = make_grid_env()
                gym_man = make_gym_env(env_ppo)
                model_man = load_model()

                obs_man = env_man.reset(
                    options={"time serie id": manual_scenario}
                )
                done_man = False
                steps_man = 0
                rho_man = []

                # Застосовуємо вибрані відключення
                for lid, active in enumerate(st.session_state.line_states):
                    if not active:
                        obs_man, _, done_man, _ = env_man.step(
                            env_man.action_space(
                                {"set_line_status": [(lid, -1)]}
                            )
                        )
                        if done_man:
                            break

                # Симулюємо ручний режим
                while not done_man and steps_man < manual_steps:
                    obs_man, _, done_man, _ = env_man.step(
                        env_man.action_space({})
                    )
                    steps_man += 1
                    active_lines = obs_man.rho[obs_man.line_status]
                    rho_man.append(
                        np.max(active_lines) if len(active_lines) > 0 else 0.0
                    )

                # ── PPO на тому самому сценарії ────────
                obs_ppo, _ = gym_man.reset(
                    options={"time serie id": manual_scenario}
                )
                obs_ppo_g2op = env_ppo.current_obs
                done_ppo = False
                steps_ppo = 0
                rho_ppo = []

                while not done_ppo and steps_ppo < manual_steps:
                    act, _ = model_man.predict(obs_ppo, deterministic=True)
                    obs_ppo, _, term, trunc, _ = gym_man.step(act)
                    obs_ppo_g2op = env_ppo.current_obs
                    done_ppo = term or trunc
                    steps_ppo += 1
                    active_lines = obs_ppo_g2op.rho[obs_ppo_g2op.line_status]
                    rho_ppo.append(
                        np.max(active_lines) if len(active_lines) > 0 else 0.0
                    )

            # ── РЕЗУЛЬТАТИ ─────────────────────────────
            st.markdown("### 📊 Результат")
            r1, r2, r3 = st.columns(3)
            r1.metric("Твій результат", f"{steps_man} кроків")
            r2.metric("PPO агент", f"{steps_ppo} кроків")

            if steps_ppo > steps_man:
                diff = steps_ppo - steps_man
                r3.metric("Переможець", "🤖 PPO",
                          delta=f"+{diff} кроків")
            elif steps_man > steps_ppo:
                diff = steps_man - steps_ppo
                r3.metric("Переможець", "👤 Ти",
                          delta=f"+{diff} кроків")
            else:
                r3.metric("Переможець", "🤝 Нічия", delta="0 кроків")

            # ── ГРАФІК ─────────────────────────────────
            st.markdown("### 📈 Завантаженість ліній")
            fig_man, ax_man = plt.subplots(figsize=(8, 3))
            fig_man.patch.set_facecolor('#1a1a2e')
            ax_man.set_facecolor('#1a1a2e')

            if rho_man:
                ax_man.plot(rho_man, color='#EF5350', lw=2,
                           label='Твоє керування')
            if rho_ppo:
                ax_man.plot(rho_ppo, color='#42A5F5', lw=2,
                           label='PPO агент')

            ax_man.axhline(y=1.0, color='#FF5722', lw=1.5,
                          ls='--', alpha=0.7, label='Межа (100%)')
            ax_man.set_xlabel('Крок', color='white')
            ax_man.set_ylabel('Завантаженість', color='white')
            ax_man.tick_params(colors='white')
            ax_man.legend(facecolor='#333', labelcolor='white')
            ax_man.grid(True, alpha=0.2)
            for sp in ax_man.spines.values():
                sp.set_color('#444')
            plt.tight_layout()
            st.pyplot(fig_man)

            # ── ТОПОЛОГІЯ ──────────────────────────────
            st.markdown("### 🗺️ Топологія")
            col_m1, col_m2 = st.columns(2)

            with col_m1:
                fig_m1 = draw_topology(
                    obs_man, env_man,
                    "Твоє керування", '#EF5350'
                )
                if done_man:
                    st.error("⚡ БЛЕКАУТ!")
                else:
                    st.info(f"👤 Ти: {steps_man} кроків")
                st.pyplot(fig_m1)

            with col_m2:
                fig_m2 = draw_topology(
                    obs_ppo_g2op, env_ppo,
                    "PPO агент", '#4CAF50'
                )
                if done_ppo:
                    st.warning("⚠️ PPO не зміг утримати")
                else:
                    st.success(f"🤖 PPO: {steps_ppo} кроків")
                st.pyplot(fig_m2)

        else:
            st.info("👈 Вибери які лінії відключити і натисни **Запустити**")
            st.markdown("""
**Як користуватись:**
1. Вибери сценарій навантаження
2. Зніми галочки з ліній які хочеш відключити
3. Натисни "Запустити ручну симуляцію"
4. Порівняй свій результат з PPO агентом

**Підказка:** спробуй відключити Лінію 17 і подивись 
що станеться — це найнавантаженіша лінія в мережі!
            """)