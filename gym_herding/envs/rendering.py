"""
2D Rendering Framework for Herding Experiment

Written by: Zahi Kakish (zmk5)

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class HerdingEnvPlotting():
    """
    Plotting methods and data for OpenAI Herding Env class

    Parameters
    ----------
    n_v : int
        Square root of the number of vertices.
    n_p : int
        The population of agents to herd.

    """
    def __init__(self, n_v, n_p):
        self.fig = None
        self.axis = None
        self.plots = None
        self._param = {
            "n_v": n_v,
            "n_p": n_p,
        }

    def create_figure(self):
        """ Create figure for rendering """
        plt.ion()
        self.fig = plt.figure(1)
        self.axis = self.fig.add_subplot(111, xlim=(0, 1), ylim=(0, 1))
        self.axis.grid(True)
        plt.xticks(np.linspace(0, 1, self._param["n_v"] + 1))
        plt.yticks(np.linspace(0, 1, self._param["n_v"] + 1))
        a_plt, = self.axis.plot([], [], 'bx', markersize=5)
        l_plt, = self.axis.plot([], [], 'r.', markersize=15)
        self.plots = [a_plt, l_plt]

    def render(self, graph, leader, is_initial=False):
        """ Render the motion of the leader and agents on the plot """
        def animate(i):
            """ Animation procedure for Fraction option """
            plot_viz_x = []
            plot_viz_y = []
            for node in graph:
                node_x, node_y = node.position
                agent_count = node.agent_count

                for _ in range(0, agent_count):
                    plot_viz_x.append(self._get_visual_position(node_x))
                    plot_viz_y.append(self._get_visual_position(node_y))

            self.plots[0].set_data(plot_viz_x, plot_viz_y)
            self.plots[1].set_data(leader.visual[0], leader.visual[1])
            return self.plots

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

    def reset(self):
        """ Reset certain mutable class properties """
        pass

    def save_render(self, file_name):
        """ Save image of the render """
        self.fig.savefig(file_name)

    def _init(self):
        """ Initial plotting of leader and agents """
        self.plots[0].set_data([], [], 'bx', markersize=5)
        self.plots[1].set_data([], [], 'r.', markersize=15)
        return self.plots

    def _get_visual_position(self, point):
        """ Get x or y coordinate for visualization """
        return point / self._param["n_v"] + np.random.uniform() / \
            self._param["n_v"]
