"""
Leader Class

Written by: Zahi Kakish (zmk5)

"""
import numpy as np


class Leader():
    """
    Leader class
    """
    def __init__(self, initial_state, n_v, pos_x, pos_y):
        self._param = {
            "n_v": n_v,
            "x": pos_x,  # initial x
            "y": pos_y,  # initial y
            "initial_state": initial_state,
        }
        self._real_position = np.array([pos_x, pos_y], dtype=np.int8)
        self._visual_position = np.array([pos_x / n_v + 1 / (n_v * 2),
                                          pos_y / n_v + 1 / (n_v * 2)])
        self._path = {
            "path": np.arange(np.power(n_v, 2)),
            "index": 0,
            "size": 0,
        }
        self.state = initial_state

    @property
    def real(self):
        """ Real Position getter property """
        return self._real_position

    @real.setter
    def real(self, pos):
        """ Real Position setter property """
        if isinstance(pos, np.ndarray):
            self._real_position = pos

        else:
            raise TypeError("Position parameter must be a np.ndarray.")

    def set_real_pos(self, ind, pos):
        """ Real Position setter for individual agent positions """
        if isinstance(pos, np.ndarray):
            self._real_position[ind, :] = pos

        else:
            raise TypeError("Position parameter must be a np.ndarray.")

    @property
    def visual(self):
        """ Visual Position Getter property """
        return self._visual_position

    @visual.setter
    def visual(self, pos):
        """ Visual Position setter property """
        if isinstance(pos, np.ndarray):
            self._visual_position = np.array(
                [pos[0] / self._param["n_v"] + 1 / (self._param["n_v"] * 2),
                 pos[1] / self._param["n_v"] + 1 / (self._param["n_v"] * 2)])

        else:
            raise TypeError("Position parameter must be a np.ndarray.")

    def set_visual_pos(self, ind, pos):
        """ Visual Position setter for individual agent positions """
        if isinstance(pos, np.ndarray):
            self._visual_position[ind, :] = np.array(
                [pos[0] / 2, pos[1] / 2]) / self._param["n_v"]

        else:
            raise TypeError("Position parameter must be a np.ndarray.")

    def set_leader_path(self, path):
        """ Set the path of states that the leader follows """
        if isinstance(path, (np.ndarray, list)):
            self._path["path"] = path
            self._path["index"] = 0
            self._path["size"] = len(path) - 1

        else:
            raise TypeError("path should be a list or np.ndarray.")

    def next_state(self):
        """ Move leader to next state in the path """
        if self._path["index"] >= self._path["size"]:
            self._path["index"] = 0

        else:
            self._path["index"] += 1

        # Set state as new step in path.
        self.state = self._path["path"][self._path["index"]]

    def reset(self):
        """ Reset certain mutable class properties """
        self._real_position = np.array(
            [self._param["x"], self._param["y"]], dtype=np.int8)
        self._visual_position = np.array(
            [self._param["x"] / self._param["n_v"] + 1 / \
             (self._param["n_v"] * 2),
             self._param["y"] / self._param["n_v"] + 1 / \
             (self._param["n_v"] * 2)])
        self._path["path"] = np.arange(np.power(self._param["n_v"], 2))
        self._path["index"] = 0
        self._path["size"] = 0

        self.state = self._param["initial_state"]