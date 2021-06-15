import os
import os.path as osp
import json
from time import time
from shutil import rmtree

import numpy as np
import numpy.random as npr
import cv2

from plot import LabelledTreePlot
from config import COLOURS, TIMES, PARAMS


class LabelledTreeAnimation(LabelledTreePlot):

    def __init__(self, n, **kwargs):
        super().__init__(n=n, **kwargs)

    def steps(self, length):
        '''
        Computes the steps for a path of a given length.
        This code is chosen to create a smooth movement effect.
        '''
        steps = np.ones((self.params['animation']['n_steps'] + 2*length - 2, 2*length - 1))
        steps = np.triu(np.cumsum(np.tril(steps), axis=0), - self.params['animation']['n_steps'] + 1)
        steps = steps[:,[2*x for x in range(length)]]/(self.params['animation']['n_steps'] + 1)
        steps = (steps**2*(1 - steps))**2
        steps = steps/np.sum(steps, axis=0, keepdims=True)

        # magnet_step is used to create the 'magnet effect' between the extra edge and the new node arriving
        magnet_step = self.params['animation']['n_steps'] - self.params['animation']['magnet']

        return steps, magnet_step

    def newframe(self, frame_time=None):
        '''
        Creates a new frame of the video.
        '''
        self.graph()
        self.savefig(save_dir=self.frames_dir, fig_time=frame_time)
        self.__figure__()

    def empty_to_tree(self):
        '''
        Creates the animation from nothing on the canvas to the tree.
        '''
        self.newframe(frame_time=self.times['start'])

        # making the nodes appear in order
        for node in sorted(self.nodes, key=lambda x:self.height(x)):
            self.node_plots[node]['plot'] = True
            parent = self.parent(node)
            if parent is not None:
                self.edge_plots[parent, node]['plot'] = True

            for i in range(self.params['animation']['n_appear']):
                ratio = (i+1)/self.params['animation']['n_appear']
                self.node_plots[node]['r'] = ratio*self.r

                ratio = i/(self.params['animation']['n_appear'] - 1)
                if parent is not None:
                    self.edge_plots[parent, node]['r'] = ratio**2*self.r

                self.newframe()

            self.newframe(frame_time=self.times['appeared'])

        # removing the unneccesary parameters
        for node in self.nodes:
            del self.node_plots[node]['r']
        for edge in self.edges:
            del self.edge_plots[edge]['r']

        self.newframe(frame_time=self.times['pause'])

    def find_leaf(self, leaves):
        '''
        Creates the effect where it finds the next leaf.
        '''
        leaf = leaves.pop()

        for _ in range(self.params['animation']['n_change'] - 1):
            self.node_plots[leaf]['colour'] = self.colours['explore']
            self.newframe(frame_time=self.times['change'])
            del self.node_plots[leaf]['colour']
            self.newframe(frame_time=self.times['change'])

        self.node_plots[leaf]['colour'] = self.colours['explore']
        self.newframe(frame_time=self.times['change'] + self.times['pause'])

        return leaf

    def path_to_leaf(self, leaf, previous_edges):
        '''
        Creates the effect where it reaches the new leaf.
        '''
        leaf_path = self.path(leaf, edge_list=True)
        leaf_path = [e for e in leaf_path if e not in previous_edges]

        for u,v in leaf_path:
            self.node_plots[u]['colour'] = self.colours['explore']
            self.newframe(frame_time=self.times['explore'])

            self.edge_plots[u,v]['colour'] = self.colours['explore']
            self.newframe(frame_time=self.times['explore'])

        self.newframe(frame_time=self.times['break'])

        return leaf_path, previous_edges + leaf_path

    def move_leaf_path(self, leaf_path, bottom, top):
        '''
        Creates the effect where the path to the leaf moves to the bottom of the frame.
        - bottom corresponds to the position at which the path will attach (bottom <= 0);
        - top corresponds to the index of the leaf of the path (top > n)
        '''
        if bottom < 0: # this is not the first path we explore
            bottom_x, bottom_y = self.node_plots[bottom + 1].get('pos', self.pos[bottom + 1])
            extra_x, extra_y = self.node_plots[bottom].get('pos', self.pos[bottom])
            self.edge_plots[bottom + 1, bottom]['pos'] = (
                np.array([bottom_x, extra_x]),
                np.array([bottom_y, extra_y])
            )

        moving_nodes = {} # a dictionary of the upcoming movements
        for i in range(len(leaf_path)):
            moving_nodes[bottom - i] = {
                'step_index' : i,
                'delta_pos' : self.pos[bottom - i] - self.pos[self.node_plots[bottom - i]['node']],
            }

            self.node_plots[bottom - i]['plot'] = True
            self.node_plots[bottom - i]['zorder'] = self.params['zorder']['moving']
            self.node_plots[bottom - i]['pos'] = self.pos[self.node_plots[bottom - i]['node']].copy()
            self.edge_plots[bottom - i, bottom - i - 1]['plot'] = True
            self.edge_plots[bottom - i, bottom - i - 1]['zorder'] = self.params['zorder']['moving']

        extra_node = bottom - len(leaf_path)
        moving_nodes[extra_node] = {
            'step_index' : len(leaf_path),
            'delta_pos' : self.pos[extra_node] - self.pos[self.node_plots[top]['node']]
                          + self.extra_edge_cut*(self.node_plots[extra_node]['group'] != 'extra'),
        }
        moving_nodes[top] = {
            'step_index' : len(leaf_path),
            'delta_pos' : self.pos[top] - self.pos[self.node_plots[top]['node']]
        }

        self.node_plots[extra_node]['plot'] = False
        self.node_plots[extra_node]['zorder'] = self.params['zorder']['moving']
        self.node_plots[extra_node]['pos'] = self.pos[self.node_plots[top]['node']].copy()
        self.node_plots[top]['plot'] = True
        self.node_plots[top]['zorder'] = self.params['zorder']['moving']
        self.node_plots[top]['pos'] = self.pos[self.node_plots[top]['node']].copy()
        
        # we now start printing the frames
        steps, magnet_step = self.steps(length=len(leaf_path) + 1)
        for step_index, step in enumerate(steps):
            for node, node_info in moving_nodes.items():
                self.node_plots[node]['pos'] += step[node_info['step_index']]*node_info['delta_pos']

            if bottom < 0: # this is not the first path we explore
                for i in range(self.params['animation']['magnet_steps']):
                    if step_index == magnet_step - i - 1:
                        next_extra_x, next_extra_y = self.node_plots[bottom].get('pos', self.pos[bottom])
                        extra_p = (i+1)/(self.params['animation']['magnet_steps'] + 1)
                        self.edge_plots[bottom + 1, bottom]['pos'] = (
                            np.array([bottom_x, extra_x*extra_p + next_extra_x*(1 - extra_p)]),
                            np.array([bottom_y, extra_y*extra_p + next_extra_y*(1 - extra_p)])
                        )
                if step_index == magnet_step:
                    # removing the unneccesary parameter
                    del self.edge_plots[bottom + 1, bottom]['pos']

            self.newframe()

        # removing the unneccesary parameters
        for node in moving_nodes:
            del self.node_plots[node]['zorder']
            if node != extra_node:
                del self.node_plots[node]['pos']
        for i in range(len(leaf_path)):
            del self.edge_plots[bottom - i, bottom - i - 1]['zorder']

        self.newframe(frame_time=self.times['pause'])

        return bottom - len(leaf_path), top + 1

    def clean(self, previous_edges, leaves):
        '''
        Creates the effect where the explored path is removed from the tree.
        '''
        node_deg = {}
        for u,v in previous_edges:
            if u in node_deg:
                node_deg[u] += 1
            else:
                node_deg[u] = 1
            if not self.degree(v):
                node_deg[v] = 0

            # if the edge was not already removed
            if not self.edge_plots[u,v].get('un_covered', False):
                self.edge_plots[u,v]['colour'] = self.colours['change']
                self.node_plots[u]['colour'] = self.colours['change']
                self.node_plots[v]['colour'] = self.colours['change']

        self.newframe(frame_time=self.times['clean'])

        # animating the path that was just explored
        for _ in range(self.params['animation']['n_clean'] - 1):
            for u,v in previous_edges:
                if not self.edge_plots[u,v].get('un_covered', False):
                    self.edge_plots[u,v]['colour'] = self.colours['explore']
                    self.node_plots[u]['colour'] = self.colours['explore']
                    self.node_plots[v]['colour'] = self.colours['explore']
            self.newframe(frame_time=self.times['clean'])

            for u,v in previous_edges:
                if not self.edge_plots[u,v].get('un_covered', False):
                    self.edge_plots[u,v]['colour'] = self.colours['change']
                    self.node_plots[u]['colour'] = self.colours['change']
                    self.node_plots[v]['colour'] = self.colours['change']
            self.newframe(frame_time=self.times['clean'])

        # marking the edges from the path as removed
        for u,v in previous_edges:
            self.edge_plots[u,v]['colour'] = self.colours['un_edge']
            self.edge_plots[u,v]['zorder'] = self.params['zorder']['un_covered']
            self.edge_plots[u,v]['un_covered'] = True

        for node, degree in node_deg.items():
            if self.degree(node) == degree: # no edges left from this node
                self.node_plots[node]['colour'] = self.colours['un_' + self.node_plots[node]['type']]
                self.node_plots[node]['zorder'] = self.params['zorder']['un_covered']
                self.node_plots[node]['un_covered'] = True
            else:
                self.node_plots[node]['colour'] = self.colours[self.node_plots[node]['type']]

        self.newframe(frame_time=self.times['moment'])

    def fade_tree(self):
        '''
        Creates the effect where the tree fades away.
        '''
        n_fade = self.params['animation']['n_fade']
        for i in range(n_fade):
            for node in self.nodes:
                self.node_plots[node]['alpha'] = (n_fade - i - 1)/n_fade
            for edge in self.edges:
                self.edge_plots[edge]['alpha'] = (n_fade - i - 1)/n_fade
            self.newframe()

    def tree_to_bij(self):
        '''
        Creates the animation from the tree to the bijection.
        '''
        leaves = sorted(list(self.leaves), reverse=True)
        previous_edges = []
        bottom = 0
        top = self.n + 1

        while leaves:
            leaf = self.find_leaf(leaves)
            leaf_path, previous_edges = self.path_to_leaf(leaf, previous_edges)
            bottom, top = self.move_leaf_path(leaf_path, bottom, top)
            self.clean(previous_edges, leaves)
        self.fade_tree()

    def find_path(self, bottoms, current_tree):
        '''
        Creates the effect where it identifies the current path of the bijection to consider.
        '''
        bottom = bottoms.pop()
        node_bottom = self.node_plots[bottom].get('node', None)
        path = [bottom]
        current_tree.append(node_bottom)
        bottom = bottoms.pop()
        node_bottom = self.node_plots[bottom].get('node', None)
        while (node_bottom is not None) & (node_bottom not in current_tree):
            path.append(bottom)
            current_tree.append(node_bottom)
            bottom = bottoms.pop()
            node_bottom = self.node_plots[bottom].get('node', None)
        if node_bottom is not None:
            bottoms.append(bottom)

        # path is now identified
        for node in path:
            self.node_plots[node]['colour'] = self.colours['explore']
            self.newframe(frame_time=self.times['explore'])
            self.edge_plots[node, node - 1]['colour'] = self.colours['explore']
            self.newframe(frame_time=self.times['explore'])

        self.newframe(frame_time=self.times['break'])

        return path, bottoms, current_tree

    def move_path_to_tree(self, path, top):
        '''
        Creates the effect moving the path from the bottom to the tree.
        '''
        moving_nodes = {} # a dictionary of the upcoming movements
        for node_index, node in enumerate(path):
            moving_nodes[node] = {
                'step_index' : node_index,
                'delta_pos' : self.pos[self.node_plots[node]['node']] - self.pos[node],
            }

            del self.node_plots[node]['colour']
            self.node_plots[node]['pos'] = self.pos[node].copy()
            self.node_plots[node]['zorder'] = self.params['zorder']['moving']
            del self.edge_plots[node, node - 1]['colour']
            self.edge_plots[node, node - 1]['zorder'] = self.params['zorder']['moving']

        moving_nodes[top] = {
                'step_index' : node_index + 1,
                'delta_pos' : self.pos[self.node_plots[top]['node']] - self.pos[top],
        }
        moving_nodes[self.node_plots[top]['node']] = {
                'step_index' : node_index + 1,
                'delta_pos' : self.pos[self.node_plots[top]['node']] - self.pos[node - 1],
        }

        self.node_plots[top]['pos'] = self.pos[top].copy()
        self.node_plots[top]['zorder'] = self.params['zorder']['moving']
        self.node_plots[self.node_plots[top]['node']]['pos'] = self.pos[node - 1].copy()
        self.edge_plots[node, self.node_plots[top]['node']] = {
            'type' : 'leaf_edge',
            'plot' : True,
            'zorder' : self.params['zorder']['moving'],
        }
        self.edge_plots[node, node - 1]['plot'] = False

        # we now start printing the frames
        steps, _ = self.steps(length=len(path) + 1)
        for step in steps:
            for node, node_info in moving_nodes.items():
                self.node_plots[node]['pos'] += step[node_info['step_index']]*node_info['delta_pos']

            self.newframe()

        # removing the unneccesary parameters
        for node in path:
            del self.node_plots[node]['pos']
            del self.node_plots[node]['zorder']
            del self.edge_plots[node, node - 1]['zorder']
            self.node_plots[node]['plot'] = False
            self.node_plots[self.node_plots[node]['node']]['plot'] = True
            self.edge_plots[node, node - 1]['plot'] = False
        del self.node_plots[top]['pos']
        del self.node_plots[top]['zorder']
        self.node_plots[top]['plot'] = False
        del self.node_plots[self.node_plots[top]['node']]['pos']
        self.node_plots[self.node_plots[top]['node']]['plot'] = True
        del self.edge_plots[node, self.node_plots[top]['node']]
        for u,v in zip(path, path[1:] + [top]):
            self.edge_plots[self.node_plots[u]['node'], self.node_plots[v]['node']]['plot'] = True

        self.newframe(frame_time=self.times['pause'])

    def bij_to_tree(self):
        '''
        Creates the animation from the bijection to the tree.
        '''
        self.newframe(frame_time=self.times['start'])
        bottoms = list(range(-self.n + 1, 1))
        tops = list(range(self.n + self.n_leaves, self.n, -1))
        current_tree = []
        while bottoms:
            path, bottoms, current_tree = self.find_path(bottoms, current_tree)
            self.move_path_to_tree(path, tops.pop())

    def tree_to_empty(self):
        '''
        Creates the animation from the tree to nothing on the canvas.
        '''
        self.newframe(frame_time=self.times['pause'])
        for i in range(self.params['animation']['n_disappear']):
            ratio = (self.params['animation']['n_disappear'] - i - 1)/self.params['animation']['n_disappear']
            for node in self.nodes:
                self.node_plots[node]['r'] = ratio*self.r
                self.node_plots[node]['pos'] = ratio**2*self.pos[node]

            self.newframe()

        # removing the unneccesary parameters
        for node in self.nodes:
            del self.node_plots[node]['r']
            del self.node_plots[node]['pos']
            self.node_plots[node]['plot'] = False
        for edge in self.edges:
            self.edge_plots[edge]['plot'] = False

        self.newframe(frame_time=self.times['end'])

    def make_frames(self, frames_dir=None):
        '''
        Creates all the frames for the global animation.
        '''
        print('Making the frames...')
        if frames_dir is not None:
            self.frames_dir = frames_dir
        if osp.exists(self.frames_dir):
            rmtree(self.frames_dir)
        os.makedirs(self.frames_dir)

        self.fig_times = {} # restarting the figure times
        print(f'** take a look in \'{self.frames_dir}\' to see the frames being made **')

        # making the frames
        self.__plots__(show_tree=False, show_bij=False)
        self.empty_to_tree()

        self.__plots__(show_tree=True, show_bij=False)
        self.tree_to_bij()

        self.__plots__(show_tree=False, show_bij=True)
        self.bij_to_tree()

        self.__plots__(show_tree=True, show_bij=False)
        self.tree_to_empty()

        # saving the times
        json.dump(self.fig_times, open(osp.join(self.frames_dir, 'times.json'), 'w'), indent=2)

    def frames_to_video(self, frames_dir=None, name='labelled-tree'):
        '''
        Transforms a directory of frames into a video.
        '''
        print('Frames created, making the video...')
        if frames_dir is None:
            frames_dir = self.frames_dir
        if not osp.exists(frames_dir):
            raise Exception(f'The directory {frames_dir} does not exist and the frames need to be computed.')

        frame_times = json.load(open(osp.join(frames_dir, 'times.json'), 'r'))

        frames = []
        for frame, time in frame_times.items():
            frames += [osp.join(frames_dir, frame)]*int(np.ceil(time*self.params['animation']['fps']))

        h, w, _ = cv2.imread(frames[0]).shape

        video_file = osp.join(self.videos_dir, name + '.avi')
        video = cv2.VideoWriter(
            video_file,
            cv2.VideoWriter_fourcc(*'XVID'),
            self.params['animation']['fps'],
            (w, h)
        )

        for frame in frames:
            image = cv2.imread(frame)
            video.write(image)

        video.release()
        cv2.destroyAllWindows()
        print(f'Video saved in \'{video_file}\'')

    def video(self, frames_dir=None, name='labelled-tree'):
        '''
        Creates the video from start to finish
        '''
        start_time = time()
        print('Creating the video')
        self.make_frames(frames_dir)
        self.frames_to_video(frames_dir, name)
        end_time = time()
        print(f'Video made in {int(end_time - start_time)}s\n')