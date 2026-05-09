# evaluate.py
import numpy as np
from src.env_setup import make_grid_env, make_gym_env
from stable_baselines3 import PPO


def run_episode_no_agent(env):
    obs = env.reset()
    done, steps, total_reward = False, 0, 0
    while not done:
        obs, reward, done, _ = env.step(env.action_space({}))
        total_reward += reward
        steps += 1
    return steps, total_reward


def run_episode_agent(gym_env, model):
    obs, _ = gym_env.reset()
    done, steps, total_reward = False, 0, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = gym_env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    return steps, total_reward


def run_episode_random(gym_env):
    obs, _ = gym_env.reset()
    done, steps, total_reward = False, 0, 0
    while not done:
        action = gym_env.action_space.sample()
        obs, reward, terminated, truncated, _ = gym_env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    return steps, total_reward


def run_episode_greedy(env):
    """Жадібний агент — відключає найперевантаженішу лінію"""
    obs = env.reset()
    done, steps, total_reward = False, 0, 0
    while not done:
        rho = obs.rho.copy()
        rho[~obs.line_status] = 0
        if np.max(rho) > 0.9:
            line_to_disconnect = np.argmax(rho)
            action = env.action_space(
                {"set_line_status": [(int(line_to_disconnect), -1)]}
            )
        else:
            action = env.action_space({})
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
    return steps, total_reward


def evaluate(model_name="ppo_grid2op_v3", n_episodes=5):
    env = make_grid_env()
    gym_env = make_gym_env(env)
    model = PPO.load(model_name, env=gym_env)

    print(f"Тестування {n_episodes} епізодів...\n")

    no_steps, no_rewards = [], []
    ag_steps, ag_rewards = [], []
    rnd_steps, rnd_rewards = [], []
    grd_steps, grd_rewards = [], []

    for i in range(n_episodes):
        s, r = run_episode_no_agent(env)
        no_steps.append(s); no_rewards.append(r)

        s, r = run_episode_agent(gym_env, model)
        ag_steps.append(s); ag_rewards.append(r)

        s, r = run_episode_random(gym_env)
        rnd_steps.append(s); rnd_rewards.append(r)

        env_g = make_grid_env()
        s, r = run_episode_greedy(env_g)
        grd_steps.append(s); grd_rewards.append(r)

        print(f"Епізод {i+1}: без агента={no_steps[-1]:4d} | "
              f"random={rnd_steps[-1]:4d} | "
              f"greedy={grd_steps[-1]:4d} | "
              f"PPO={ag_steps[-1]:4d}")

    print("\n=== ПІДСУМОК ===")
    print(f"{'Метод':<15} {'Кроки (сер.)':<15} {'Винагорода (сер.)'}")
    print(f"{'Без агента':<15} {np.mean(no_steps):<15.0f} {np.mean(no_rewards):.0f}")
    print(f"{'Random agent':<15} {np.mean(rnd_steps):<15.0f} {np.mean(rnd_rewards):.0f}")
    print(f"{'Greedy agent':<15} {np.mean(grd_steps):<15.0f} {np.mean(grd_rewards):.0f}")
    print(f"{'PPO агент':<15} {np.mean(ag_steps):<15.0f} {np.mean(ag_rewards):.0f}")

    print(f"\nPPO vs без агента: {np.mean(ag_steps)/np.mean(no_steps):.2f}x")
    print(f"PPO vs random:     {np.mean(ag_steps)/np.mean(rnd_steps):.2f}x")
    print(f"PPO vs greedy:     {np.mean(ag_steps)/np.mean(grd_steps):.2f}x")

    worst_no = np.argmin(no_steps)
    print(f"\n=== КРИТИЧНИЙ СЦЕНАРІЙ (епізод {worst_no+1}) ===")
    print(f"Без агента: {no_steps[worst_no]} кроків")
    print(f"Greedy:     {grd_steps[worst_no]} кроків")
    print(f"PPO агент:  {ag_steps[worst_no]} кроків")
    ratio = ag_steps[worst_no] / max(no_steps[worst_no], 1)
    print(f"Покращення PPO vs без агента: {ratio:.0f}x")

    return {
        'no_agent': {'steps': no_steps, 'rewards': no_rewards},
        'random':   {'steps': rnd_steps, 'rewards': rnd_rewards},
        'greedy':   {'steps': grd_steps, 'rewards': grd_rewards},
        'ppo':      {'steps': ag_steps,  'rewards': ag_rewards},
    }


if __name__ == "__main__":
    evaluate()