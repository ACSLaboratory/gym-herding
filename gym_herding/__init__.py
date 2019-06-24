"""
Register New Herding Environment
"""
from gym.envs.registration import register


register(
    id="Herding-v0",
    entry_point="gym_herding.envs:HerdingEnv",
)
