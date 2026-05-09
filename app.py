# app.py — інтерактивний дашборд керування енергомережею
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from env_setup import make_grid_env, make_gym_env
from stable_baselines3 import PPO

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
    model = PPO.load("ppo_grid2op_v3", env=gym_env)
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
    st.subheader("🔌 Відключити лінію вручну")
    line_names = [f"Лінія {i}" for i in range(20)]
    disabled_line = st.selectbox("Вибери лінію для відключення",
                                  ["(не відключати)"] + line_names)

    st.divider()
    max_steps = st.slider("Кроків симуляції", 20, 200, 100, step=10)
    run_button = st.button("▶️ Запустити симуляцію", type="primary",
                           use_container_width=True)

tab1, tab2, tab3 = st.tabs(["🔬 Симуляція", "📊 Порівняння сценаріїв", "🏗️ Архітектура"])

with tab1:
    if run_button:
        model = load_model()

        with st.spinner("Симуляція..."):
            env_no  = make_grid_env()
            env_ag  = make_grid_env()
            env_grd = make_grid_env()
            gym_env = make_gym_env(env_ag)

            manual_off = None
            if disabled_line != "(не відключати)":
                manual_off = int(disabled_line.split(" ")[1])

            # ── БЕЗ АГЕНТА ────────────────────────────
            steps_no, rho_history_no = [], []
            obs_no = env_no.reset(options={"time serie id": scenario})
            done_no = False
            if manual_off is not None:
                obs_no, _, done_no, _ = env_no.step(
                    env_no.action_space({"set_line_status": [(manual_off, -1)]})
                )
            for _ in range(max_steps):
                if done_no: break
                obs_no, _, done_no, _ = env_no.step(env_no.action_space({}))
                steps_no.append(len(steps_no))
                active = obs_no.rho[obs_no.line_status]
                rho_history_no.append(np.max(active) if len(active) > 0 else 0.0)

            # ── PPO АГЕНТ ──────────────────────────────
            steps_ag, rho_history_ag = [], []
            actions_count_ag = 0
            event_log = []
            obs_gym, _ = gym_env.reset(options={"time serie id": scenario})
            obs_ag = env_ag.current_obs
            done_ag = False
            if manual_off is not None:
                obs_ag, _, done_ag, _ = env_ag.step(
                    env_ag.action_space({"set_line_status": [(manual_off, -1)]})
                )
            prev_status_ag = obs_ag.line_status.copy()
            for _ in range(max_steps):
                if done_ag: break
                act, _ = model.predict(obs_gym, deterministic=True)
                obs_gym, _, term, trunc, _ = gym_env.step(act)
                obs_ag = env_ag.current_obs
                done_ag = term or trunc
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
            obs_grd = env_grd.reset(options={"time serie id": scenario})
            done_grd = False
            if manual_off is not None:
                obs_grd, _, done_grd, _ = env_grd.step(
                    env_grd.action_space({"set_line_status": [(manual_off, -1)]})
                )
            prev_status_grd = obs_grd.line_status.copy()
            for _ in range(max_steps):
                if done_grd: break
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

        st.subheader("📊 Результати симуляції")
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
            if done_ag:
                st.warning("⚠️ Агент не зміг утримати")
            else:
                st.success(f"✅ PPO: {survived_ag} кроків!")
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
        st.image("topology.png", caption="Приклад топології IEEE 14-bus")

with tab2:
    st.subheader("📊 Порівняння всіх 20 сценаріїв")

    @st.cache_resource
    def run_all_scenarios():
        rows = []
        progress = st.progress(0, text="Тестування сценаріїв...")

        for sid in range(20):
            env_base = make_grid_env()
            obs = env_base.reset(options={"time serie id": sid})
            done, steps_no = False, 0
            while not done and steps_no < 300:
                obs, _, done, _ = env_base.step(env_base.action_space({}))
                steps_no += 1

            env_ag  = make_grid_env()
            gym_ag  = make_gym_env(env_ag)
            model_s = PPO.load("ppo_grid2op_v3", env=gym_ag)
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
                "Сценарій": f"#{sid}",
                "Тип": label,
                "Без агента": steps_no,
                "Greedy": steps_grd,
                "PPO агент": steps_ag,
                "PPO vs без агента": f"{steps_ag/max(steps_no,1):.1f}x",
                "PPO vs Greedy": f"{steps_ag/max(steps_grd,1):.1f}x",
            })
            progress.progress((sid+1)/20,
                              text=f"Тестування сценарію {sid+1}/20...")

        progress.empty()
        return rows

    if st.button("🔄 Запустити порівняння всіх сценаріїв",
                 type="primary", use_container_width=True):
        with st.spinner("Це займе ~2 хвилини..."):
            rows = run_all_scenarios()

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        fig_all, ax_all = plt.subplots(figsize=(12, 5))
        fig_all.patch.set_facecolor('#1a1a2e')
        ax_all.set_facecolor('#1a1a2e')
        x = np.arange(20)
        w = 0.25
        no_vals  = [r["Без агента"] for r in rows]
        grd_vals = [r["Greedy"] for r in rows]
        ppo_vals = [r["PPO агент"] for r in rows]
        ax_all.bar(x - w, no_vals,  w, label='Без агента',
                   color='#EF5350', alpha=0.85)
        ax_all.bar(x,     grd_vals, w, label='Greedy',
                   color='#FF9800', alpha=0.85)
        ax_all.bar(x + w, ppo_vals, w, label='PPO агент',
                   color='#42A5F5', alpha=0.85)
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
    else:
        st.info("Натисни кнопку щоб запустити порівняння всіх 20 сценаріїв. "
                "Займе ~2 хвилини.")
        
with tab3:
    st.subheader("🏗️ Архітектура системи")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 👁️ Observation Space")
        st.markdown("Агент отримує вектор стану розміром **114 параметрів**:")
        obs_data = {
            "Параметр": ["rho", "p_or", "gen_p", "load_p", "topo_vect"],
            "Опис": [
                "Завантаженість ліній (% від максимуму)",
                "Активна потужність на початку лінії",
                "Генерація активної потужності",
                "Споживання навантажень",
                "Топологічний вектор вузлів"
            ],
            "Розмір": [20, 20, 6, 11, 57],
        }
        st.dataframe(obs_data, use_container_width=True, hide_index=True)

        st.markdown("### ⚡ Action Space")
        st.markdown("Агент може виконати **101 дискретну дію**:")
        st.markdown("""
        - **1** дія — нічого не робити (do nothing)
        - **40** дій — підключити/відключити кожну з 20 ліній
        - **60** дій — перемикання шин у підстанціях
        """)

    with col_r:
        st.markdown("### 🎯 Reward Function")
        st.latex(r"R(t) = \alpha \cdot S(t) - \beta \cdot O(t) - \gamma \cdot L(t) + \delta \cdot B(t)")

        reward_data = {
            "Компонент": ["S(t)", "O(t)", "L(t)", "B(t)"],
            "Назва": [
                "Стабільність",
                "Штраф перевантаження",
                "Штраф блекауту",
                "Бонус з'єднаності"
            ],
            "Коефіцієнт": ["α = 1.0", "β = 2.0", "γ = 100.0", "δ = 0.5"],
            "Опис": [
                "+1 за кожен крок без аварії",
                "-2 × сума перевантажень > 90%",
                "-100 при блекауті",
                "+0.5 × частка активних ліній"
            ]
        }
        st.dataframe(reward_data, use_container_width=True, hide_index=True)

        st.markdown("### 🧠 Алгоритм PPO")
        st.markdown("""
        **Proximal Policy Optimization (PPO)** обраний через:
        - Стабільність навчання на дискретних діях
        - Ефективність у задачах керування
        - Кліпування градієнтів запобігає деградації політики
        """)

        ppo_params = {
            "Гіперпараметр": [
                "learning_rate", "n_steps", "batch_size",
                "n_epochs", "ent_coef", "clip_range", "gamma"
            ],
            "Значення": [
                "3×10⁻⁴", "2048", "64", "10", "0.01", "0.2", "0.99"
            ],
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

    st.divider()
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

    st.divider()
    st.markdown("### 🕸️ Енергомережа як мультиграф")

    col_graph_l, col_graph_r = st.columns(2)

    with col_graph_l:
        st.markdown("""
**Формальне визначення:**

Енергомережа моделюється як мультиграф **G = (V, E)** де:

- **V** — множина вузлів (|V| = 14):
  - 6 вузлів-генераторів (джерела енергії)
  - 11 вузлів-навантажень (споживачі)
  - підстанції (точки з'єднання)
- **E** — множина ребер (|E| = 20):
  - лінії передачі електроенергії
  - допускаються кратні ребра між вузлами
- **w(e)** — вага ребра = завантаженість лінії ρ ∈ [0, ∞)

**Задача керування:**

Знайти оптимальну підмножину активних ребер **E' ⊆ E** після 
аварійного відключення, що мінімізує функцію:

$$\\min_{E' \\subseteq E} \\sum_{e \\in E'} \\max(\\rho_e - 0.9, 0)$$

за умови збереження зв'язності графу G' = (V, E').
        """)

    with col_graph_r:
        st.markdown("""
**Зв'язок з RL:**

Задача керування топологією формулюється як **MDP (Марківський процес прийняття рішень)**:

| Компонент MDP | Відповідність у задачі |
|---------------|----------------------|
| **Стан S** | Вектор завантаженості ребер + топологія |
| **Дія A** | Зміна підмножини активних ребер E' |
| **Винагорода R** | Функція стабільності мережі |
| **Перехід P** | Динаміка потокорозподілу |
| **Горизонт T** | До блекауту або кінця епізоду |

**Чому мультиграф, а не простий граф?**

У реальних енергосистемах між двома підстанціями може існувати 
кілька паралельних ліній передачі. Мультиграф дозволяє точно 
моделювати цю структуру і керувати кожною лінією незалежно.
        """)

    st.divider()
    st.markdown("### 🔄 Алгоритм керування топологією")
    st.markdown("""
**Цикл прийняття рішень агента на кожному кроці t:**
    """)

    col_alg1, col_alg2, col_alg3, col_alg4 = st.columns(4)
    with col_alg1:
        st.info("**1. Спостереження**\n\nАгент отримує вектор стану s(t) розміром 114: завантаженість ліній, генерація, навантаження, топологія")
    with col_alg2:
        st.warning("**2. Прийняття рішення**\n\nPPO нейромережа обирає дію a(t) з 101 можливих: яку лінію підключити або відключити")
    with col_alg3:
        st.error("**3. Зміна топології**\n\nСередовище застосовує дію: змінює підмножину активних ребер E' і перераховує потокорозподіл")
    with col_alg4:
        st.success("**4. Отримання винагороди**\n\nАгент отримує R(t) = α·S - β·O - γ·L + δ·B і переходить до кроку t+1")