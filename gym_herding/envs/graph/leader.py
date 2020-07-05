"""
Leader Class.

Written by: Zahi Kakish (zmk5)

"""
from typing import Union
from typing import List

import numpy as np


class Leader():
    """
    Leader class for Graph.
    """

    def __init__(self, initial_state: int, n_v: int, x: int, y: int) -> None:
        """Initialize the Leader class."""
        self._param = {
            'n_v': n_v,
            'x': x,  # initial x
            'y': y,  # initial y
            'initial_state': initial_state,
        }
        self._real_position = np.array([x, y], dtype=np.int8)

        # Visual position converts matrix coordinates to cartesian coordinates
        # if (i, j) are the matrix indexes, then the cartesian coordinates are:
        #
        #     x = j + offset
        #     y = (n_v - i) - offset
        self.offset: float = 0.5
        self._visual_position = np.array(
            [y + self.offset,  # x in cartesian space
             (n_v - x) - self.offset])  # y in cartesian space

        self.path = {
            'path': np.arange(np.power(n_v, 2)),
            'index': 0,
            'size': 0,
        }
        self.state = initial_state

    @property
    def real(self) -> np.ndarray:
        """Real Position getter property."""
        return self._real_position

    @real.setter
    def real(self, pos: np.ndarray) -> None:
        """Real Position setter property."""
        if isinstance(pos, np.ndarray):
            self._real_position = pos

        else:
            raise TypeError('Position parameter must be a np.ndarray.')

    def set_real_pos(self, ind: int, pos: np.ndarray) -> None:
        """Real Position setter for individual agent positions."""
        if isinstance(pos, np.ndarray):
            self._real_position[ind, :] = pos

        else:
            raise TypeError('Position parameter must be a np.ndarray.')

    @property
    def visual(self) -> np.ndarray:
        """Visual Position Getter property."""
        return self._visual_position

    @visual.setter
    def visual(self, pos: np.ndarray) -> None:
        """Visual Position setter property."""
        if isinstance(pos, np.ndarray):
            self._visual_position = np.array(
                [pos[1] + self.offset,
                 (self._param['n_v'] - pos[0]) - self.offset])

        else:
            raise TypeError('Position parameter must be a np.ndarray.')

    def set_visual_pos(self, ind: int, pos: np.ndarray) -> None:
        """Visual Position setter for individual agent positions."""
        if isinstance(pos, np.ndarray):
            self._visual_position[ind, :] = np.array(
                [pos[0] / 2, pos[1] / 2]) / self._param['n_v']

        else:
            raise TypeError('Position parameter must be a np.ndarray.')

    def set_leader_path(self, path: Union[np.ndarray, List[int]]) -> None:
        """Set the path of states that the leader follows."""
        if isinstance(path, (np.ndarray, list)):
            self.path['path'] = path
            self.path['index'] = 0
            self.path['size'] = len(path) - 1

        else:
            raise TypeError('path should be a list or np.ndarray.')

    def next_state(self) -> None:
        """Move leader to next state in the path."""
        if self.path['index'] >= self.path['size']:
            self.path['index'] = 0

        else:
            self.path['index'] += 1

        # Set state as new step in path.
        self.state = self.path['path'][self.path['index']]

    def reset(self) -> None:
        """Reset certain mutable class properties."""
        self._real_position = np.array(
            [self._param['x'], self._param['y']], dtype=np.int8)
        self._visual_position = np.array(
            [self._param['y'] + self.offset,
             (self._param['n_v'] - self._param['x']) - self.offset])
        self.path['path'] = np.arange(np.power(self._param['n_v'], 2))
        self.path['index'] = 0
        self.path['size'] = 0

        self.state = self._param['initial_state']
