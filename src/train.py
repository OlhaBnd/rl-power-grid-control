# train.py
from src.env_setup import make_grid_env, make_gym_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class TrainingCallback(BaseCallback):
    """Логує прогрес навчання"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self):
        if self.locals.get("dones", [False])[0]:
            info = self.locals.get("infos", [{}])[0]
            ep_info = info.get("episode")
            if ep_info:
                self.episode_rewards.append(ep_info["r"])
                self.episode_lengths.append(ep_info["l"])
        return True


def train(timesteps=300_000, model_name="ppo_grid2op_v3"):
    print("=" * 50)
    print("Навчання PPO з власною Reward Function")
    print(f"Кроків: {timesteps:,}")
    print("=" * 50)

    env = make_grid_env()
    gym_env = make_gym_env(env)

    print("\nПростір спостережень (Observation Space):")
    print(f"  Розмір: {gym_env.observation_space.shape}")
    print(f"  Параметри: rho, p_or, gen_p, load_p, topo_vect")

    print("\nПростір дій (Action Space):")
    print(f"  Розмір: {gym_env.action_space.n} дискретних дій")
    print(f"  Тип: перемикання статусів ліній")

    print("\nФункція винагороди (Reward Function):")
    print("  R(t) = α·S(t) - β·O(t) - γ·L(t) + δ·B(t)")
    print("  α=1.0 (стабільність), β=2.0 (перевантаження)")
    print("  γ=100.0 (блекаут), δ=0.5 (з'єднаність)")

    # Гіперпараметри PPO
    hyperparams = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "gamma": 0.99,
    }

    print("\nГіперпараметри PPO:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")

    model = PPO(
        "MlpPolicy",
        gym_env,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        **hyperparams
    )

    callback = TrainingCallback()

    print("\nПочинаємо навчання...")
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        tb_log_name="PPO_v3_custom_reward"
    )

    model.save(model_name)
    print(f"\nМодель збережена: {model_name}.zip")

    if callback.episode_rewards:
        print(f"Середня винагорода (останні 10 епізодів): "
              f"{np.mean(callback.episode_rewards[-10:]):.2f}")
        print(f"Середня довжина епізоду: "
              f"{np.mean(callback.episode_lengths[-10:]):.0f} кроків")

    return model


if __name__ == "__main__":
    train()