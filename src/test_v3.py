from src.env_setup import make_grid_env, make_gym_env
from stable_baselines3 import PPO
import numpy as np

env = make_grid_env()
gym_env = make_gym_env(env)
model = PPO.load('ppo_grid2op_v3', env=gym_env)

results = []
for i in range(5):
    obs, _ = gym_env.reset()
    done, steps = False, 0
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = gym_env.step(act)
        done = term or trunc
        steps += 1
    results.append(steps)
    print(f'Епізод {i+1}: {steps} кроків')

print(f'Середнє: {np.mean(results):.0f} кроків')