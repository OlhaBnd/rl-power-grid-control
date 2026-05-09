# env_setup.py
import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Parameters import Parameters
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, DiscreteActSpace
from reward import CustomReward


def make_grid_env(scenario_id=None):
    """Створює grid2op середовище з власною функцією винагороди"""
    param = Parameters()
    param.NO_OVERFLOW_DISCONNECTION = False
    param.NB_TIMESTEP_OVERFLOW_ALLOWED = 2

    env = grid2op.make(
        "l2rpn_case14_sandbox",
        backend=LightSimBackend(),
        param=param,
        reward_class=CustomReward  # власна reward function
    )

    if scenario_id is not None:
        env.reset(options={"time serie id": scenario_id})

    return env


def make_gym_env(env):
    """Обгортає grid2op env у gym-сумісний формат для PPO"""
    gym_env = GymEnv(env)
    gym_env.observation_space = BoxGymObsSpace(
        env.observation_space,
        attr_to_keep=["rho", "p_or", "gen_p", "load_p", "topo_vect"]
    )
    gym_env.action_space = DiscreteActSpace(
        env.action_space,
        attr_to_keep=["set_line_status"]
    )
    return gym_env