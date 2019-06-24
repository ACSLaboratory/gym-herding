"""
Herding Environment Parameter Class

Written by: Zahi Kakish (zmk5)
"""
import numpy as np


class HerdingEnvParameters():
    """
    Parameters for use in the OpenAI Herding Environment.

    Parameters
    ----------
    n_v : int
        Square root of the number of vertices.
    n_p : int
        The number of agents to herd
    weights : list
        The weight applied to the population distribution within the state
        space.

    """
    def __init__(self, n_v, n_p, weights):
        self.iter = 0
        self.max_iter = 10000
        self.n_v = n_v
        self.n_p = n_p

        # Set weights
        if isinstance(weights, (list, np.ndarray)):
            if len(weights) != 2:
                raise ValueError("Can only be two weight values.")

            self.weights = weights

        else:
            self.weights = [1/10, 9/10]

        self.dist = {
            "target": 0,
            "initial": 0,
        }

        self.extra = {
            "init_leader_pos": np.zeros(2, dtype=np.int8),
            "t": 0,
            "dt": 0.1,
            "jump_weight": 0.1,
            "leader_motion_moves_agents": False,
        }

    def set_agents_distribution(self, val, dist):
        """ Set the herding agents distribution in space. """
        if isinstance(val, np.ndarray):
            if val.shape != (self.n_v, self.n_v):
                raise ValueError("Distribution must be n_v x n_v")

            # Set distribution as value
            self.dist[dist] = val

        else:
            raise TypeError("Distribution must be a np.ndarray.")

    def set_leader_position(self, i, j):
        """ Set an initial postion for the leader. """
        self.extra["init_leader_postion"][0] = i
        self.extra["init_leader_position"][1] = j

    def add_extra_parameter(self, key, val):
        """ Add an extra parameter, if needed."""
        self.extra[key] = val
