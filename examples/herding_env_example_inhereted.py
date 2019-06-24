"""
OpenAI Herding Environment Example

This example script shows how to use the gym with your own rewards and
observation structures using inheretance.

Written by: Zahi Kakish (zmk5)

"""
import numpy as np
from gym_herding.envs.parameters import HerdingEnvParameters
from herding_env_ihnerited_class import HerdingEnvInheritanceExample


# Set up experimental parameters and variables.
n_v = 2  # Square root of the number of graph vertices you wish to use.
n_p = 1000  # Agent population
weights = [2/10, 8/10]
done = False

hep = HerdingEnvParameters(n_v, n_p, weights)
hep.set_agents_distribution(
    np.array([[1, 0],
              [0, 1]], dtype=np.float32), "target")
hep.set_agents_distribution(
    np.array([[0.1, 0.4],
              [0.4, 0.1]], dtype=np.float32), "initial")
hep.max_iter = 1000
hep.extra["jump_weight"] = 0.1
hep.extra["leader_motion_moves_agents"] = False

# Create the environment with your new parameters.
env = HerdingEnvInheritanceExample(hep)

while not done:
    action = np.random.randint(0, 5)
    obs, reward, done, info = env.step(action)
