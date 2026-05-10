#попередній розрахунок результатів для демо
import numpy as np
import pandas as pd
import json
import sys
import os

# Додаємо кореневу папку до шляху
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env_setup import make_grid_env, make_gym_env
from stable_baselines3 import PPO


def run_benchmark(model_path="models/ppo_grid2op_v3", n_scenarios=20, max_steps=300):
    """Запускає бенчмарк всіх сценаріїв і зберігає результати"""
    print("Запуск бенчмарку всіх сценаріїв...")
    rows = []

    for sid in range(n_scenarios):
        print(f"Сценарій {sid+1}/{n_scenarios}...", end=" ")

        # Без агента
        env_no = make_grid_env()
        obs = env_no.reset(options={"time serie id": sid})
        done, steps_no = False, 0
        while not done and steps_no < max_steps:
            obs, _, done, _ = env_no.step(env_no.action_space({}))
            steps_no += 1

        # PPO агент
        env_ag = make_grid_env()
        gym_ag = make_gym_env(env_ag)
        model = PPO.load(model_path, env=gym_ag)
        obs_gym, _ = gym_ag.reset(options={"time serie id": sid})
        done, steps_ag = False, 0
        while not done and steps_ag < max_steps:
            act, _ = model.predict(obs_gym, deterministic=True)
            obs_gym, _, term, trunc, _ = gym_ag.step(act)
            done = term or trunc
            steps_ag += 1

        # Greedy
        env_grd = make_grid_env()
        obs_grd = env_grd.reset(options={"time serie id": sid})
        done, steps_grd = False, 0
        while not done and steps_grd < max_steps:
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

        # Класифікація сценарію
        if steps_no < 10:
            scenario_type = "критичний ⚠️"
        elif steps_no < 100:
            scenario_type = "важкий"
        else:
            scenario_type = "стабільний"

        rows.append({
            "scenario_id": sid,
            "type": scenario_type,
            "no_agent": steps_no,
            "greedy": steps_grd,
            "ppo": steps_ag,
            "ppo_vs_no": round(steps_ag / max(steps_no, 1), 2),
            "ppo_vs_greedy": round(steps_ag / max(steps_grd, 1), 2),
        })
        print(f"без агента={steps_no} | greedy={steps_grd} | PPO={steps_ag}")

    # Зберігаємо результати
    df = pd.DataFrame(rows)
    df.to_csv("outputs/benchmark_results.csv", index=False)

    # Зберігаємо як JSON для швидкого завантаження
    with open("outputs/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\n=== ПІДСУМОК ===")
    print(f"Сер. без агента: {df['no_agent'].mean():.0f} кроків")
    print(f"Сер. greedy:     {df['greedy'].mean():.0f} кроків")
    print(f"Сер. PPO:        {df['ppo'].mean():.0f} кроків")
    print(f"PPO vs без агента: {df['ppo_vs_no'].mean():.2f}x")
    print(f"\nЗбережено: outputs/benchmark_results.csv")
    return df


if __name__ == "__main__":
    run_benchmark()