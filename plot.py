import os
import os.path as osp
from time import time

import numpy as np
import numpy.random as npr
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Ellipse, Rectangle, Polygon
from matplotlib.patheffects import Normal, Stroke

from model import LabelledTree
from config import COLOURS, TIMES, PARAMS


class LabelledTreePlot(LabelledTree):

    def __init__(self,
                 images_dir='images',
                 frames_dir='images/frames',
                 videos_dir='videos',
                 colours=COLOURS,
                 times=TIMES,
                 params=PARAMS,
                 plot_numbers=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.images_dir = images_dir
        if not osp.exists(self.images_dir):
            os.makedirs(self.images_dir)
        self.frames_dir = frames_dir
        if not osp.exists(self.frames_dir):
            os.makedirs(self.frames_dir)
        self.videos_dir = videos_dir
        if not osp.exists(self.videos_dir):
            os.makedirs(self.videos_dir)
        self.plot_numbers = plot_numbers

        self.colours = colours
        self.times = times
        self.params = params

        self.__attrs__()

    def __attrs__(self):
        '''
        Sets important attributes to the class:
        - self.r: the basis radius for the spheres and sticks;
        - self.index: the index of the image saved, useful when saving multiple images;
        - self.fig_times: the dictionary of times when saving multiple figures;
        - self.x_shade: the basis of the displacement of the shade on the x axis;
        - self.y_shade: the basis of the displacement of the shade on the y axis;
        - self.min_frame: the position of the central part of the frame;
        - self.max_frame: the position of the outter part of the frame;
        - self.middle_frame: the middle position of the frame;
        - self.extra_edge_cut: the length of the extra edge corresponding to the end of Cayley's sequence.
        '''
        self.r = 1/(2*self.n)
        self.index = 0
        self.fig_times = {}
        self.x_shade = self.params['shade']['shift']*np.cos(self.params['shade']['angle'])*self.r
        self.y_shade = self.params['shade']['shift']*np.sin(self.params['shade']['angle'])*self.r
        self.min_frame = 1 + self.params['frame']['min']*self.r
        self.max_frame = 1 + self.params['frame']['max']*self.r
        self.middle_frame = (self.max_frame + self.min_frame)/2
        self.extra_edge_cut = np.array([-self.params['animation']['extra_edge']*self.max_frame/self.n, 0])

        self.__positions__() # computes the positions of the nodes
        self.__plots__() # computes the information about the plots
        self.__figure__() # creates a new figure

    def __positions__(self):
        '''
        Sets the positions of the nodes, using kamada kawai layout.
        Create self.pos, a dictionary of positions such that:
        - for 1 <= node <= n, self.pos[node] refers to the position of the node;
        - for n + 1 <= i <= n + n_leaves, self.pos[i] refers to the position of the i-th leaf;
        - for 1 - n <= i <= 0, self.pos[i] refers to the position of the -i-th node in the cayley sequence;
        '''
        self.pos = {}
        for n in self.nodes:
            h = self.height(n)
            self.pos[n] = (h*npr.rand(), h) # forces the nodes to be ordered by height
        extra_edges = []
        for j in range(2*self.degree(self.root)): # extra nodes and edges to highlight the root in the tree representation
            self.pos[self.n + 1 + j] = (0, -(1 + j))
            extra_edges.append((self.root, self.n + 1 + j))

        # positioning the nodes of the tree
        self.pos = nx.kamada_kawai_layout(nx.from_edgelist(list(self.edges) + extra_edges), pos=self.pos)
        for j in range(2*self.degree(self.root)): # remove the extra nodes
            del self.pos[self.n + 1 + j]

        # positioning the nodes of the cayley sequence at the bottom of the frame
        for i in range(self.n):
            self.pos[-i] = np.array([- self.middle_frame + 2*self.max_frame*i/self.n, - self.middle_frame])
        self.pos[- self.n + 1] += self.extra_edge_cut

        # positioning the leaves at the top of the frame
        for i in range(self.n_leaves):
            self.pos[self.n + 1 + i] = np.array([- self.middle_frame + 2*self.max_frame*i/self.n, self.middle_frame])

    def __plots__(self, show_tree=True, show_bij=True):
        '''
        Creates the dictionaries used to plot the nodes and edges.
        For each key of the dictionary, the corresponding values contain basic information:
        - the type of node: root, node, or leaf;
        - the group it belongs to: tree of bij; belonging to the bij group means that the node belongs to the frame;
        - if it should be plotted or not.
        On top of these arguments, extra ones can be added to specify more information about the key (see plot_nodes and plot_edges).
        '''
        self.node_plots = {}
        self.edge_plots = {}

        for n in self.nodes:
            if n == self.root:
                self.node_plots[n] = {
                    'type' : 'root',
                    'group' : 'tree',
                    'plot' : show_tree,
                }
            elif n in self.leaves:
                self.node_plots[n] = {
                    'type' : 'leaf',
                    'group' : 'tree',
                    'plot' : show_tree,
                }
            else:
                self.node_plots[n] = {
                    'type' : 'node',
                    'group' : 'tree',
                    'plot' : show_tree,
                }

        for u,v in self.edges:
            if v in self.leaves:
                self.edge_plots[u,v] = {
                    'type' : 'leaf_edge',
                    'group' : 'tree',
                    'plot' : show_tree,
                }
            else:
                self.edge_plots[u,v] = {
                    'type' : 'edge',
                    'group' : 'tree',
                    'plot' : show_tree,
                }

        # a special function is used for the nodes of the bijection
        self.__bij__(show_bij)

    def __bij__(self, show_bij):
        '''
        Completes the dictionaries created in __plots__.
        Computes the information for all the nodes of the group bij, corresponding to the representation of the Cayley's bijection.
        '''
        edges = []
        neg_index = 0
        for index_leaf, leaf in enumerate(self.leaves):
            self.node_plots[self.n + 1 + index_leaf] = {
                'type' : 'leaf',
                'group' : 'bij',
                'plot' : show_bij,
                'node' : leaf, # an extra argument to refer to the corresponding node in the tree
            }

            leaf_path = self.path(leaf, edge_list=True)
            if index_leaf != 0:
                repeating = 'repeating_'
            else:
                repeating = ''

            for u,v in leaf_path:
                if (u,v) not in edges:
                    edges.append((u,v))

                    self.node_plots[neg_index] = {
                        'type' : repeating + self.node_plots[u]['type'],
                        'group' : 'bij',
                        'plot' : show_bij,
                        'node' : u, # an extra argument to refer to the corresponding node in the tree
                    }
                    repeating = ''
                    if v != leaf:
                        self.edge_plots[neg_index, neg_index - 1] = {
                            'type' : 'edge',
                            'group' : 'bij',
                            'plot' : show_bij,
                        }
                    else:
                        self.edge_plots[neg_index, neg_index - 1] = {
                            'type' : 'leaf_edge',
                            'group' : 'bij',
                            'plot' : show_bij,
                        }

                    neg_index -= 1

        self.node_plots[neg_index] = {
            'type' : 'node',
            'group' : 'extra', # a special group for the extra node added at the end of Cayley's sequence
            'plot' : False,
        }


    def __figure__(self):
        '''
        Creates a new figure.
        '''
        if hasattr(self, 'fig'):
            plt.close('all')

        self.fig, self.ax = plt.subplots(figsize=(self.params['other']['figure_size']*self.max_frame,
                                                  self.params['other']['figure_size']*self.max_frame))
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.set_axis_off()
        self.ax.set_xlim(-self.max_frame, self.max_frame)
        self.ax.set_ylim(-self.max_frame, self.max_frame)
        self.__frame__()

    def __frame__(self):
        # Creates the frame
        self.ax.add_patch(
            Rectangle(
                (-self.max_frame*1.1, -self.max_frame*1.1),
                2.2*self.max_frame,
                2.2*self.max_frame,
                lw=0,
                color=self.params['frame']['colour'],
                zorder=self.params['zorder']['shade']
            )
        )
        self.ax.add_patch(
            Rectangle(
                (-self.min_frame*self.params['frame']['border'], -self.min_frame*self.params['frame']['border']),
                2*self.min_frame*self.params['frame']['border'],
                2*self.min_frame*self.params['frame']['border'],
                lw=0,
                color=self.params['frame']['border_colour'],
                zorder=self.params['zorder']['shade'] + .1
            )
        )
        self.ax.add_patch(
            Rectangle(
                (-self.min_frame, -self.min_frame),
                2*self.min_frame,
                2*self.min_frame,
                lw=0,
                color='white',
                zorder=self.params['zorder']['shade'] + .2
            )
        )
        self.ax.add_patch(
            Rectangle(
                (-self.min_frame, -self.min_frame),
                2*self.min_frame,
                2*self.min_frame,
                lw=0,
                color='white',
                zorder=self.params['zorder']['shade'] + .3,
                alpha=self.params['frame']['alpha_canvas']
            )
        )
        # Creates the top, bottom, left and right polygons of the frame
        top = np.array(
            [
                [-self.max_frame, self.max_frame],
                [-self.min_frame, self.min_frame],
                [self.min_frame, self.min_frame],
                [self.max_frame, self.max_frame]
            ]
        )
        bottom = np.array(
            [
                [-self.max_frame, -self.max_frame],
                [-self.min_frame, -self.min_frame],
                [self.min_frame, -self.min_frame],
                [self.max_frame, -self.max_frame]
            ]
        )
        left = np.array(
            [
                [-self.max_frame, self.max_frame],
                [-self.min_frame, self.min_frame],
                [-self.min_frame, -self.min_frame],
                [-self.max_frame, -self.max_frame]
            ]
        )
        right = np.array(
            [
                [self.max_frame, self.max_frame],
                [self.min_frame, self.min_frame],
                [self.min_frame, -self.min_frame],
                [self.max_frame, -self.max_frame]
            ]
        )
        # Creates the alpha values of the polygons of the frame
        alpha_top = max(0, -np.sin(self.params['shade']['angle']))
        alpha_bottom = max(0, np.sin(self.params['shade']['angle']))
        alpha_left = max(0, np.cos(self.params['shade']['angle']))
        alpha_right = max(0, -np.cos(self.params['shade']['angle']))
        # Adds the shade on the polygons of the frame, to create 3D effect
        self.ax.add_patch(Polygon(top,
                                  color='black',
                                  alpha=self.params['frame']['black_shade']*alpha_top,
                                  zorder=self.params['zorder']['shade'] + .4,
                                  lw=0))
        self.ax.add_patch(Polygon(bottom,
                                  color='black',
                                  alpha=self.params['frame']['black_shade']*alpha_bottom,
                                  zorder=self.params['zorder']['shade'] + .4,
                                  lw=0))
        self.ax.add_patch(Polygon(left,
                                  color='black',
                                  alpha=self.params['frame']['black_shade']*alpha_left,
                                  zorder=self.params['zorder']['shade'] + .4,
                                  lw=0))
        self.ax.add_patch(Polygon(right,
                                  color='black',
                                  alpha=self.params['frame']['black_shade']*alpha_right,
                                  zorder=self.params['zorder']['shade'] + .4,
                                  lw=0))
        self.ax.add_patch(Polygon(top,
                                  color='white',
                                  alpha=self.params['frame']['white_shade']*alpha_bottom*3,
                                  zorder=self.params['zorder']['shade'] + .4,
                                  lw=0))
        self.ax.add_patch(Polygon(bottom,
                                  color='white',
                                  alpha=self.params['frame']['white_shade']*alpha_top*3,
                                  zorder=self.params['zorder']['shade'] + .4,
                                  lw=0))
        self.ax.add_patch(Polygon(left,
                                  color='white',
                                  alpha=self.params['frame']['white_shade']*alpha_right*3,
                                  zorder=self.params['zorder']['shade'] + .4,
                                  lw=0))
        self.ax.add_patch(Polygon(right,
                                  color='white',
                                  alpha=self.params['frame']['white_shade']*alpha_left*3,
                                  zorder=self.params['zorder']['shade'] + .4,
                                  lw=0))

    def resample(self):
        '''
        Resample the class.
        '''
        super().resample()
        self.__attrs__()

    def plot_sphere(self, X, Y, r, colour, zorder, alpha, node):
        '''
        Plots the sphere of the node at position X,Y, with radius r.
        '''
        self.ax.add_patch(
            Circle(
                (X + self.x_shade, Y + self.y_shade),
                r,
                color=self.params['shade']['colour'],
                zorder=self.params['zorder']['shade'] + .25,
                alpha=alpha
            )
        )
        self.ax.add_patch(
            Circle(
                (X, Y),
                r,
                color=colour,
                zorder=zorder + .5,
                alpha=alpha
            )
        )

        # creating the shade on the sphere
        self.ax.add_patch(
            Wedge(
                (X, Y),
                r,
                self.params['shade']['angle']*180/np.pi - 90,
                self.params['shade']['angle']*180/np.pi + 90,
                color='black',
                zorder=zorder + .6,
                alpha=self.params['shade']['alpha']*alpha
            )
        )
        for s in range(self.params['shade']['n'] + 1):
            self.ax.add_patch(
                Ellipse(
                    (X, Y),
                    2*(self.params['shade']['start'] + s*(1 - self.params['shade']['start'])/self.params['shade']['n'])*r,
                    2*r, self.params['shade']['angle']*180/np.pi,
                    color=colour,
                    zorder=zorder + .7,
                    alpha=alpha/(1+s)
                )
            )

        # plots the node number next to the sphere
        if self.plot_numbers:
            text = self.ax.text(
                X + r,
                Y + r,
                node,
                fontsize=500*r,
                fontweight='normal',
                zorder=zorder + .8,
                color='white',
                alpha=alpha
            )
            text.set_path_effects(
                [
                    Stroke(linewidth=100*r, foreground='black'),
                    Stroke(linewidth=50*r, foreground=colour),
                    Normal()
                ]
            )

    def plot_stick(self, X, Y, r_tuple, colour, zorder, alpha):
        '''
        Plots the stick between X and Y.
        '''
        r_u, r_v, r = r_tuple
        dist_to_u = r_u*self.params['animation']['distance_to_node']
        dist_to_v = r_v*self.params['animation']['distance_to_node']
        base_width = r*self.params['other']['ratio_point_to_inch']
        edge_ratio = self.params['animation']['edge_ratio']

        dX = X[1] - X[0]
        dY = Y[1] - Y[0]
        edge_length = (dX**2 + dY**2)**.5
        if edge_length > dist_to_u + dist_to_v: # only plots the stick if X and Y are sufficiently far away
            dX, dY = dX/edge_length, dY/edge_length

            X0, Y0 = X[0] + dist_to_u*dX, Y[0] + dist_to_u*dY
            X1, Y1 = X[1] - dist_to_v*dX, Y[1] - dist_to_v*dY

            # the shade of the stick
            self.ax.plot(
                [X0 + self.x_shade, X1 + self.x_shade],
                [Y0 + self.y_shade, Y1 + self.y_shade],
                self.params['shade']['colour'],
                solid_capstyle='round',
                linewidth=base_width*edge_ratio,
                zorder=self.params['zorder']['shade'] + .25,
                alpha=alpha
            )
            # the stick
            self.ax.plot(
                [X0, X1],
                [Y0, Y1],
                colour,
                solid_capstyle='round',
                linewidth=base_width*edge_ratio,
                zorder=zorder,
                alpha=alpha
            )

            # creating the shade on the rounded part of the stick
            self.ax.add_patch(
                Wedge(
                    (X0, Y0),
                    r*edge_ratio,
                    self.params['shade']['angle']*180/np.pi - 90,
                    self.params['shade']['angle']*180/np.pi + 90,
                    color='black',
                    zorder=zorder + .1,
                    alpha=alpha*self.params['shade']['alpha']
                )
            )
            self.ax.add_patch(
                Wedge(
                    (X1, Y1),
                    r*edge_ratio,
                    self.params['shade']['angle']*180/np.pi - 90,
                    self.params['shade']['angle']*180/np.pi + 90,
                    color='black',
                    zorder=zorder + .1,
                    alpha=alpha*self.params['shade']['alpha']
                )
            )
            self.ax.add_patch(
                Ellipse(
                    (X0, Y0),
                    2*self.params['shade']['start']*r*edge_ratio,
                    2*r*edge_ratio,
                    self.params['shade']['angle']*180/np.pi,
                    color=colour,
                    zorder=zorder + .2,
                    alpha=alpha
                )
            )
            self.ax.add_patch(
                Ellipse(
                    (X1, Y1),
                    2*self.params['shade']['start']*r*edge_ratio,
                    2*r*edge_ratio,
                    self.params['shade']['angle']*180/np.pi,
                    color=colour,
                    zorder=zorder + .2,
                    alpha=alpha
                )
            )
            self.ax.plot(
                [X0, X1],
                [Y0, Y1],
                colour,
                solid_capstyle='butt',
                linewidth=base_width*edge_ratio,
                zorder=zorder + .3,
                alpha=alpha
            )

            # creating the shade on the length of the stick
            oX = -dY
            oY = dX
            scalar = oX*np.cos(self.params['shade']['angle']) + oY*np.sin(self.params['shade']['angle'])
            if scalar > 0: # orienting according to the shade direction
                oX, oY = -oX, -oY

            if dX == 0:
                if dY > 0:
                    edge_angle = np.pi/2
                else:
                    edge_angle = -np.pi/2
            elif dX > 0:
                edge_angle = np.arctan(dY/dX)
            else:
                edge_angle = np.pi + np.arctan(dY/dX)

            shade_height = r*edge_ratio*(1-1/(4 - 3*np.cos(self.params['shade']['angle'] - edge_angle)**2))/2
            
            self.ax.add_patch(
                Rectangle(
                    (X0 - edge_ratio*r*oX, Y0 - edge_ratio*r*oY),
                    edge_length - dist_to_u - dist_to_v,
                    shade_height,
                    edge_angle*180/np.pi,
                    color='black',
                    zorder=zorder + .4,
                    alpha=alpha*self.params['shade']['alpha'],
                    lw=0
                )
            )

    def plot_nodes(self):
        '''
        Plot the nodes as defined in self.node_plot.
        Uses a set ground values if not given in the dictionary.
        '''
        for node, node_plot in self.node_plots.items():
            if node_plot['plot']:
                X, Y = node_plot.get('pos', self.pos[node])
                r = node_plot.get('r', self.r)
                colour = node_plot.get('colour', self.colours[node_plot['type']])
                zorder = node_plot.get('zorder', self.params['zorder']['fixed'])
                alpha = node_plot.get('alpha', 1)
                node = node_plot.get('node', node)
                self.plot_sphere(X, Y, r, colour, zorder, alpha, node)

    def plot_edges(self):
        '''
        Plot the edges as defined in self.edge_plot.
        Uses a set ground values if not given in the dictionary.
        '''
        for (u,v), edge_plot in self.edge_plots.items():
            if edge_plot['plot']:
                X, Y = edge_plot.get(
                    'pos',
                    (
                        np.array([
                            self.node_plots[u].get('pos', self.pos[u])[0],
                            self.node_plots[v].get('pos', self.pos[v])[0]
                        ]),
                        np.array([
                            self.node_plots[u].get('pos', self.pos[u])[1],
                            self.node_plots[v].get('pos', self.pos[v])[1]
                        ])
                    )
                )
                r_tuple = (
                    self.node_plots[u].get('r', self.r),
                    self.node_plots[v].get('r', self.r),
                    edge_plot.get('r', self.r)
                )
                colour = edge_plot.get('colour', self.colours[edge_plot['type']])
                zorder = edge_plot.get('zorder', self.params['zorder']['fixed'])
                alpha = edge_plot.get('alpha', 1)
                self.plot_stick(X, Y, r_tuple, colour, zorder, alpha)

    def graph(self):
        '''
        Plots the graph.
        '''
        self.plot_nodes()
        self.plot_edges()

    def savefig(self, name=None, fig_time=None, save_dir=None):
        '''
        Save the current version of the figure
        '''
        if name is None:
            name = str(self.index)
            self.index += 1
        name += '.png'

        if fig_time is None:
            fig_time = 1/self.params['animation']['fps']
        self.fig_times[name] = fig_time

        if save_dir is None:
            save_dir = self.images_dir

        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        self.fig.savefig(osp.join(save_dir, name))

        return osp.join(save_dir, name)

    def tree(self, name='tree', show_tree=True, show_bij=True, **kwargs):
        '''
        Save the image of the tree.
        '''
        start_time = time()
        print('Plotting the tree')
        self.__plots__(show_tree, show_bij)
        self.graph()
        print(f'Figure saved in \'{self.savefig(name=name)}\'')
        self.__figure__()
        end_time = time()
        print(f'Tree plotted in {int(end_time - start_time)}s\n')