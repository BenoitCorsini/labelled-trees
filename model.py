import numpy as np
import numpy.random as npr


class LabelledTree(object):

    def __init__(self, n=None, nodes=None, edges=None, seed=None, **kwargs):
        self.n = n
        self._nodes = nodes
        self._edges = edges
        self.seed = seed
        npr.seed(self.seed)

        if (self.n is None) & (self._nodes is None) & (self._edges is None):
            raise Exception('No input in \'{}\' class'.format(self.__class__.__name__))

        if self.n is None:
            if self._nodes is None:
                self.n = np.size(self._edges) + 1
            else:
                self.n = np.size(self._nodes) + 1

        self.__nodes__()
        self.__edges__()
        self.__tree__()

    def __nodes__(self):
        if self._nodes is None:
            self._nodes = 1 + npr.randint(self.n, size=self.n - 1)
        self._nodes = self._nodes.astype(int)
        self.nodes = np.arange(1, self.n + 1)

        if self.n != np.size(self._nodes) + 1:
            raise Exception('\'nodes\' in \'{}\' of the wrong size: {}'.format(self.__class__.__name__, self._nodes))
        elif (np.max(self._nodes) > self.n) or (np.min(self._nodes) < 1):
            raise Exception('\'nodes\' in \'{}\' has wrong entries: {}'.format(self.__class__.__name__, self._nodes))

    def __edges__(self):
        if self._edges is None:
            self._edges = 1 + np.argsort(npr.rand(self.n - 1))
        self._edges = self._edges.astype(int)

        if self.n != np.size(self._edges) + 1:
            raise Exception('\'edges\' in \'{}\' of the wrong size: {}'.format(self.__class__.__name__, self._edges))
        if np.any(self._edges - 1 != np.argsort(np.argsort(self._edges))):
            raise Exception('\'edges\' in \'{}\' is not a permutation: {}'.format(self.__class__.__name__, self._edges))

    def __tree__(self):
        self.leaves = np.arange(1, self.n + 1)
        self.leaves = self.leaves[np.isin(self.leaves, self._nodes, invert=True)]
        self.root = self._nodes[0]

        self.n_leaves = len(self.leaves)
        self.n_central_nodes = self.n - self.n_leaves

        self.edges = {}
        self._path = {self.root : [self.root]}
        self.children = {}
        tree_nodes = [self._nodes[0]]
        leaf_index = 0
        for i in range(self.n - 1):
            x = self._nodes[i]
            y = self._nodes[(i+1) % (self.n - 1)]
            e = self._edges[i]
            if y in tree_nodes:
                y = self.leaves[leaf_index]
                leaf_index += 1
            else:
                tree_nodes.append(y)

            self.edges[x,y] = e
            self._path[y] = self._path[x] + [y]
            if x not in self.children:
                self.children[x] = []
            self.children[x].append(y)

    def resample(self):
        self._nodes = None
        self._edges = None
        if self.seed is not None:
            self.seed += 1
        self.__nodes__()
        self.__edges__()
        self.__tree__()        

    def degree(self, n):
        return len(self.children.get(n, []))

    def height(self, n):
        return len(self._path[n]) - 1

    def parent(self, n):
        if n == self.root:
            return None
        else:
            return self._path[n][-2]

    def path(self, n, edge_list=False):
        if edge_list:
            return list(zip(self._path[n][:-1], self._path[n][1:]))
        else:
            return self._path[n]

    def __getitem__(self, key):
        return list(zip(self._nodes, self._edges))[key]

    def __iter__(self):
        return iter(zip(self._nodes, self._edges))

    def __call__(self, key):
        return self._path[key]