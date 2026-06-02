"""
Unit-тести для функції винагороди CustomReward
Запуск: python -m pytest test_reward.py -v
"""

import pytest
from unittest.mock import MagicMock
import numpy as np


# ---------------------------------------------------------------------------
# Мок-реалізація CustomReward (незалежна від Grid2Op)
# Відтворює логіку з reward.py для ізольованого тестування
# ---------------------------------------------------------------------------

class CustomRewardMock:
    """Спрощена копія логіки CustomReward для тестування."""

    ALPHA = 1.0    # стабільність
    BETA = 2.0     # штраф перевантаження
    GAMMA = 100.0  # штраф блекауту
    DELTA = 0.5    # бонус з'єднаності
    OVERLOAD_THRESHOLD = 0.9

    def compute(self, rho: np.ndarray, is_blackout: bool, n_active_lines: int, n_total_lines: int) -> float:
        """
        rho            — масив завантаженості ліній (0.0 – 1.0+)
        is_blackout    — True якщо стався блекаут
        n_active_lines — кількість активних ліній
        n_total_lines  — загальна кількість ліній
        """
        if is_blackout:
            return -self.GAMMA

        S = self.ALPHA * 1.0
        O = self.BETA * float(np.sum(np.maximum(rho - self.OVERLOAD_THRESHOLD, 0.0)))
        B = self.DELTA * (n_active_lines / n_total_lines)

        return S - O + B


reward_fn = CustomRewardMock()


# ---------------------------------------------------------------------------
# Тести
# ---------------------------------------------------------------------------

class TestCustomReward:

    def test_no_overload_no_blackout(self):
        """Нормальна робота мережі — очікується позитивна винагорода."""
        rho = np.array([0.5, 0.6, 0.4, 0.7, 0.3])  # всі < 0.9
        result = reward_fn.compute(rho, is_blackout=False, n_active_lines=5, n_total_lines=5)
        assert result > 0, "При нормальній роботі винагорода має бути позитивною"

    def test_blackout_gives_large_penalty(self):
        """Блекаут — найбільший штраф, незалежно від стану ліній."""
        rho = np.array([0.1, 0.1, 0.1])
        result = reward_fn.compute(rho, is_blackout=True, n_active_lines=0, n_total_lines=3)
        assert result == -100.0, f"Штраф за блекаут має бути -100, отримано {result}"

    def test_overload_reduces_reward(self):
        """Перевантажені лінії зменшують винагороду."""
        rho_normal = np.array([0.5, 0.5, 0.5])
        rho_overloaded = np.array([1.1, 1.2, 0.5])  # дві лінії перевантажені
        r_normal = reward_fn.compute(rho_normal, False, 3, 3)
        r_overloaded = reward_fn.compute(rho_overloaded, False, 3, 3)
        assert r_overloaded < r_normal, "Перевантаження має зменшувати винагороду"

    def test_overload_threshold_boundary(self):
        """Лінія рівно на межі 90% — штрафу ще немає."""
        rho = np.array([0.9])
        result = reward_fn.compute(rho, is_blackout=False, n_active_lines=1, n_total_lines=1)
        # O = 2.0 * max(0.9 - 0.9, 0) = 0
        assert result > 0, "Завантаженість 90% не має давати штраф"

    def test_overload_above_threshold(self):
        """Лінія вище 90% — штраф нараховується."""
        rho_border = np.array([0.9])
        rho_over = np.array([1.0])
        r_border = reward_fn.compute(rho_border, False, 1, 1)
        r_over = reward_fn.compute(rho_over, False, 1, 1)
        assert r_over < r_border, "Завантаженість >90% має штрафуватись"

    def test_connectivity_bonus(self):
        """Більше активних ліній — вища винагорода."""
        rho = np.array([0.5] * 20)
        r_full = reward_fn.compute(rho, False, n_active_lines=20, n_total_lines=20)
        r_partial = reward_fn.compute(rho, False, n_active_lines=10, n_total_lines=20)
        assert r_full > r_partial, "Більше активних ліній має збільшувати винагороду"

    def test_blackout_overrides_normal_state(self):
        """Блекаут дає -100 навіть якщо лінії не перевантажені."""
        rho = np.zeros(20)  # всі лінії ненавантажені
        result = reward_fn.compute(rho, is_blackout=True, n_active_lines=20, n_total_lines=20)
        assert result == -100.0

    def test_reward_components_sum(self):
        """Перевірка що формула R = S - O + B правильно рахується вручну."""
        rho = np.array([1.0, 0.5])   # одна лінія перевантажена на 0.1
        # S = 1.0
        # O = 2.0 * (1.0 - 0.9) = 0.2
        # B = 0.5 * (2/2) = 0.5
        # R = 1.0 - 0.2 + 0.5 = 1.3
        result = reward_fn.compute(rho, False, n_active_lines=2, n_total_lines=2)
        assert abs(result - 1.3) < 1e-9, f"Очікувалось 1.3, отримано {result}"