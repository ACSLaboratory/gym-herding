"""
2D Histogram Rendering View for Herding Experiment.

Written by: Zahi Kakish (zmk5)

"""
from typing import List
from typing import TypeVar

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches
import matplotlib.path as mplpath

from matplotlib.animation import FuncAnimation
from gym_herding.envs.graph.graph import NodeGraph
from gym_herding.envs.graph.leader import Leader


# Used for documentation purposes only
PlotType = TypeVar('PlotType')


class HerdingEnvHistogram():
    """
    Histogram plotting methods and data for OpenAI Herding Env class

    Parameters
    ----------
    n_v : int
        Square root of the number of vertices.
    n_p : int
        The population of agents to herd.

    """

    def __init__(self, n_v: int, n_p: int) -> None:
        """Initialize the HerdingEnvHistorgram Class."""
        self.fig: plt.Figure = None
        self.axis = None
        self.patch: mplpatches.PathPatch = None
        self.barpath: mplpath.Path = None
        self._param: dict = {
            'n_v': n_v,
            'n_p': n_p,
            'n_d': np.ndarray,
            'bins': np.ndarray,
        }
        self._box: dict = {
            'left': np.ndarray,
            'right': np.ndarray,
            'bottom': np.ndarray,
            'top': np.ndarray,
            'verts': np.ndarray,
            'nverts': int,
            'nrects': int,
            'codes': np.ndarray,
        }

    def create_figure(self, graph: NodeGraph, leader: Leader) -> None:
        """Create figure for rendering."""
        # TODO: Need to add way of seeing where the leader is.
        self._initial_boxes(graph)
        self._fill_boxes()

        plt.ion()
        self.fig = plt.figure(1)
        self.axis = self.fig.add_subplot(111, xlim=(0, 1), ylim=(0, 1))
        self.barpath = mplpath.Path(self._box['verts'], self._box['codes'])
        self.patch = mplpatches.PathPatch(
            self.barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
        self.axis.add_patch(self.patch)

        self.axis.set_xlim(self._box['left'][0], self._box['right'][-1])
        self.axis.set_ylim(0, 1)  # should always be from 0 to 1 bc pop fracs

    def render(self, graph: NodeGraph, leader: Leader,
               is_initial: bool = False) -> None:
        """Render the motion of the leader and agents on the plot."""
        def animate(i) -> List[mplpatches.PathPatch]:
            # simulate new data coming in
            self._param['n_d'] = graph.distribution.current
            self._param['bins'] = np.arange(
                self._param['n_v'] * self._param['n_v'])
            self._box['top'] = self._box['bottom'] + self._param['n_d']
            self._box['verts'][1::5, 1] = self._box['top']
            self._box['verts'][2::5, 1] = self._box['top']
            return [self.patch, ]

        # Animate and draw new positions.
        if is_initial:
            anim = FuncAnimation(
                self.fig, animate, frames=1,
                init_func=self._init, repeat=False, blit=False)

        else:
            anim = FuncAnimation(
                self.fig, animate, frames=1, repeat=False,
                blit=False)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def reset(self) -> None:
        """Reset certain mutable class properties."""
        pass

    def save_render(self, file_name: str) -> None:
        """Save image of the render."""
        self.fig.savefig(file_name)

    def _init(self) -> List[mplpatches.PathPatch]:
        """Initial plotting of leader and agents."""
        return [self.patch, ]

    def _initial_boxes(self, graph: NodeGraph) -> None:
        """Create initial boxes."""
        self._param['n_d'] = graph.distribution.current
        self._param['bins'] = np.arange(self._param['n_v'] * self._param['n_v'])
        self._box['left'] = np.array(self._param['bins'][:-1])
        self._box['right'] = np.array(self._param['bins'][1:])
        self._box['bottom'] = np.zeros(len(self._box['left']))
        self._box['top'] = self._box['bottom'] + self._param['n_d']
        self._box['nrects'] = len(self._box['left'])

    def _fill_boxes(self) -> None:
        """Fill initial boxes."""
        self._box['nverts'] = self._box['nrects'] * (1 + 3 + 1)
        self._box['verts'] = np.zeros((self._box['nverts'], 2))
        self._box['codes'] = np.ones(self._box['nverts'], int) * mplpath.Path.LINETO
        self._box['codes'][0::5] = mplpath.Path.MOVETO
        self._box['codes'][4::5] = mplpath.Path.CLOSEPOLY

        self._box['verts'][0::5, 0] = self._box['left']
        self._box['verts'][0::5, 1] = self._box['bottom']

        self._box['verts'][1::5, 0] = self._box['left']
        self._box['verts'][1::5, 1] = self._box['top']

        self._box['verts'][2::5, 0] = self._box['right']
        self._box['verts'][2::5, 1] = self._box['top']

        self._box['verts'][3::5, 0] = self._box['right']
        self._box['verts'][3::5, 1] = self._box['bottom']
