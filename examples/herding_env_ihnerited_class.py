"""
Example Inherited Herding Environment
"""
import numpy as np
from gym_herding.envs.herding_env import HerdingEnv


class HerdingEnvInheritanceExample(HerdingEnv):
    """
    Something
    """
    def __init__(self, hep):
        super().__init__(hep)

    def step(self, action):
        """ Executes the given action """
        # Get new leader state and position from action
        new_lx, new_ly, new_ls, is_out_of_bounds = \
            self._graph.convert_action_to_node_info(self._leader.state, action)

        # If action takes leader out of bounds, change info state to say that
        # occured.
        self.var["is_out_of_bounds"] = is_out_of_bounds

        # Apply changes to leaders position and state onto the class
        self._move_leader(new_ls, new_lx, new_ly)

        # Apply repulsive effect of leader in new location on those local
        # herding agents.
        if self.param.extra["leader_motion_moves_agents"]:
            self._move_herding_agents()

        # Apply repulsive effect of leader in new location on those local
        # herding agents IF action STAY is used.
        else:
            if action == 4:  # Stay
                self._move_herding_agents()

        # Get newest observation of the state space.
        obs = self._get_observation()

        # Get reward for action
        if self.param.extra["leader_motion_moves_agents"]:
            reward = self._get_reward()

        else:
            reward = self._get_reward_stay(action)

        done = False

        # Write info
        info = {
            "is_out_of_bounds": self.var["is_out_of_bounds"],
            "leader_state": self._leader.state,
            "leader_pos": [new_lx, new_ly],
        }

        # Check for end of episode
        if self.param.iter == self.param.max_iter:
            done = True

        else:
            self.param.iter += 1

        return (obs, reward, done, info)

    def _get_reward_stay(self, action):
        """ Reward Test Cases """
        if action == 4:
            return -1 * np.sum(np.power(
                self._graph.distribution.current - \
                    self._graph.distribution.target, 2))

        # return -1
        return -1 * np.sum(np.power(
            self._graph.distribution.current - \
                self._graph.distribution.target, 2))

    def _get_reward(self):
        """ Reward Test Cases """
        return -1 * np.sum(np.power(
            self._graph.distribution.current - \
                self._graph.distribution.target, 2))

    def _get_observation(self):
        """ Get the calculated observation of environment state """
        return self._graph.distribution.current
