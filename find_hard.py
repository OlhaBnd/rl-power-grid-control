import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Parameters import Parameters

param = Parameters()
param.NO_OVERFLOW_DISCONNECTION = False
param.NB_TIMESTEP_OVERFLOW_ALLOWED = 2

env = grid2op.make("l2rpn_case14_sandbox",
                   backend=LightSimBackend(), param=param)

print("Шукаємо важкі сценарії...")
for scenario_id in range(20):
    obs = env.reset(options={"time serie id": scenario_id})
    done = False
    steps = 0
    while not done and steps < 200:
        obs, _, done, _ = env.step(env.action_space({}))
        steps += 1
    print(f"Сценарій {scenario_id:2d}: впав на кроці {steps:4d} {'← ВАЖКИЙ!' if steps < 20 else ''}")