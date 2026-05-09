# reward.py — власна функція винагороди
import numpy as np
from grid2op.Reward import BaseReward


class CustomReward(BaseReward):
    """
    Власна функція винагороди для керування топологією енергомережі.
    
    Математична модель:
        R(t) = α·S(t) - β·O(t) - γ·L(t) + δ·B(t)
    
    де:
        S(t) — винагорода за стабільність мережі
        O(t) — штраф за перевантаження ліній
        L(t) — штраф за блекаут
        B(t) — бонус за збереження з'єднаності
    """

    def __init__(self):
        super().__init__()
        # Коефіцієнти функції винагороди
        self.alpha = 1.0    # вага стабільності
        self.beta  = 2.0    # вага штрафу за перевантаження
        self.gamma = 100.0  # вага штрафу за блекаут
        self.delta = 0.5    # вага бонусу за з'єднаність

    def initialize(self, env):
        self.reward_min = -self.gamma
        self.reward_max = self.alpha + self.delta

    def __call__(self, action, env, has_error, is_done,
                 is_illegal, is_ambiguous):

        if is_done:
            # Блекаут — максимальний штраф
            return -self.gamma

        obs = env.get_obs()

        # S(t) — базова винагорода за виживання
        s_t = self.alpha

        # O(t) — штраф за перевантаження
        # Рахуємо суму перевантажень по всіх активних лініях
        overloaded = obs.rho[obs.line_status] - 0.9
        overloaded = np.clip(overloaded, 0, None)  # тільки > 90%
        o_t = self.beta * np.sum(overloaded)

        # B(t) — бонус за з'єднаність мережі
        # Більше активних ліній = краще
        connectivity = np.sum(obs.line_status) / len(obs.line_status)
        b_t = self.delta * connectivity

        reward = s_t - o_t + b_t

        return float(reward)