import numpy as np
import numpy.random as npr


class LabelledTree(object):

    def __init__(self, n, seed=None, **kwargs):
        self.n = n
        self.seed = seed

        self.__tree__()

    def __tree__(self):
        '''
        Creates the Cayley sequence and the corresponding line-breaking tree.
        Contains the following arguments:
        - self.cayley: a sequence of numbers in [n]^(n-1);
        - self.leaves: the sequence of leaves of the tree;
        - self.root: the root of the tree;
        - self.n_leaves: the number of leaves;
        - self.n_central_nodes: the number of nodes that are not leaves;
        - self.edges: a list of the edges, ordered as (parent, child);
        - self.paths: a dicionary where self.paths[node] is the path from the root to the node;
        - self.children: a dictionary where self.children[node] is the list of children of the node.
        '''
        npr.seed(self.seed)
        self.cayley = 1 + npr.randint(self.n, size=self.n - 1)
        self.leaves = np.arange(1, self.n + 1)
        self.leaves = self.leaves[np.isin(self.leaves, self.cayley, invert=True)]
        self.root = self.cayley[0]

        self.n_leaves = len(self.leaves)
        self.n_central_nodes = self.n - self.n_leaves

        self.edges = []
        self.paths = {self.root : [self.root]}
        self.children = {}
        tree_nodes = [self.cayley[0]] # the nodes already explored
        leaf_index = 0 # the index of the next leaf to consider
        for i in range(self.n - 1):
            x = self.cayley[i]
            y = self.cayley[(i+1) % (self.n - 1)]
            if y in tree_nodes:
                y = self.leaves[leaf_index]
                leaf_index += 1
            else:
                tree_nodes.append(y)

            self.edges.append((x,y))
            self.paths[y] = self.paths[x] + [y]
            if x not in self.children:
                self.children[x] = []
            self.children[x].append(y)

    def __iter__(self):
        '''
        When iterating through the tree, simply go through the values 1 to n.
        '''
        return iter(range(1, self.n + 1))

    def resample(self):
        '''
        Resample a labelled tree of the same size.
        '''
        if self.seed is not None:
            self.seed += 1
        self.__tree__()        

    def degree(self, n):
        '''
        Computes the degree of the node n.
        '''
        return len(self.children.get(n, []))

    def height(self, n):
        '''
        Computes the height of the node n.
        '''
        return len(self.paths[n]) - 1

    def parent(self, n):
        '''
        Computes the parent of the node n.
        '''
        if n == self.root:
            return None
        else:
            return self.paths[n][-2]

    def path(self, n, edge_list=False):
        '''
        Computes the path from the root to n.
        If edge_list = False, returns a list of the nodes.
        If edge_list = True, returns a list of the edges.
        '''
        if edge_list:
            return list(zip(self.paths[n][:-1], self.paths[n][1:]))
        else:
            return self.paths[n]