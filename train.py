# train.py
from env_setup import make_grid_env, make_gym_env
from stable_baselines3 import PPO


def train(timesteps=200_000, model_name="ppo_grid2op_v2"):
    env = make_grid_env()
    gym_env = make_gym_env(env)

    model = PPO(
        "MlpPolicy",
        gym_env,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.01,
    )

    print(f"Навчання PPO: {timesteps:,} кроків...")
    model.learn(total_timesteps=timesteps)
    model.save(model_name)
    print(f"Модель збережена: {model_name}.zip")
    return model


if __name__ == "__main__":
    train()