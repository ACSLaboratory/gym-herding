"""
OpenAI Herding Environment Example

This example script shows how to use the default herding gym environment.

Written by: Zahi Kakish (zmk5)

"""
import gym

import numpy as np

from gym_herding.envs.utils.parameters import HerdingEnvParameters


def main():
    """Run example script using OpenAI initializer."""
    env = gym.make('gym_herding:Herding-v0')

    # Set up experimental parameters and variables.
    n_v = 6  # Square root of the number of graph vertices you wish to use.
    n_p = 1000  # Agent population
    weights = [2/10, 8/10]
    done = False

    hep = HerdingEnvParameters(n_v, n_p, weights)
    hep.set_agents_distribution(
        np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]], dtype=np.float32), 'target')
    hep.set_agents_distribution(
        np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.2]], dtype=np.float32), 'initial')
    hep.max_iter = 1000
    hep.extra['jump_weight'] = 0.1
    hep.extra['leader_motion_moves_agents'] = False

    # Initialize the environment with your new parameters.
    env.initialize(hep)

    # Set observation and reward
    env._get_observation = lambda: env.graph.distribution.current
    env._get_reward = lambda: -1

    while not done:
        # Generate action
        action = np.random.randint(0, 5)

        # Render the environment
        env.render()

        # Increment one step
        obs, reward, done, info = env.step(action)


if __name__ == '__main__':
    main()
