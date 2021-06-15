import argparse

from config import COLOURS, TIMES, PARAMS
from animate import LabelledTreeAnimation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic saving parameters
    parser.add_argument('--name', type=str, default='labelled-tree',
        help='the name of the image and/or video')
    parser.add_argument('--images_dir', type=str, default='images',
        help='where to save the images')
    parser.add_argument('--frames_dir', type=str, default='images/frames',
        help='where to save the frames for the videos')
    parser.add_argument('--videos_dir', type=str, default='videos',
        help='where to save the videos')

    # what types of results to save
    parser.add_argument('--plot_tree', type=int, default=1,
        help='if the tree should be plotted or not (0 = not saved; 1 = saved)')
    parser.add_argument('--animate', type=int, default=1,
        help='if the animation should be created or not (0 = not saved; 1 = saved)')

    # parameters of the tree
    parser.add_argument('--n', type=int, default=10,
        help='the size of the tree')
    parser.add_argument('--seed', type=int, default=None,
        help='the seed for reproducibility')

    # the colours of the root
    parser.add_argument('--root_colour', type=str, default=None,
        help='the main colour of the root')
    parser.add_argument('--repeating_root_colour', type=str, default=None,
        help='the colour of the root when repeated in the bijection')
    parser.add_argument('--un_root_colour', type=str, default=None,
        help='the colour of the root when completely uncovered in the tree')
    # the colours of the leaves
    parser.add_argument('--leaf_colour', type=str, default=None,
        help='the main colour of the leaves')
    parser.add_argument('--un_leaf_colour', type=str, default=None,
        help='the colour of the leaves when uncovered in the tree')
    # the colours of the nodes
    parser.add_argument('--node_colour', type=str, default=None,
        help='the main colour of the remaining nodes')
    parser.add_argument('--repeating_node_colour', type=str, default=None,
        help='the colour of the nodes when repeated in the bijection')
    parser.add_argument('--un_node_colour', type=str, default=None,
        help='the colour of the nodes when completely uncovered in the tree')
    # the colours of the edges
    parser.add_argument('--edge_colour', type=str, default=None,
        help='the main colour of the edges')
    parser.add_argument('--leaf_edge_colour', type=str, default=None,
        help='the colour of the edges connected to a leaf')
    parser.add_argument('--un_edge_colour', type=str, default=None,
        help='the colour of the edges when uncovered in the tree')
    # the colours of the animation effects
    parser.add_argument('--explore_colour', type=str, default=None,
        help='the colour when exploring subpath of the tree or the bijection')
    parser.add_argument('--change_colour', type=str, default=None,
        help='the colour when transforming uncovered nodes and edges')

    kwargs = vars(parser.parse_args())

    # changing the defined colours
    for element in COLOURS:
        if kwargs[element + '_colour'] is not None:
            COLOURS[element] = kwargs[element + '_colour']

    # adding the general parameters
    kwargs['colours'] = COLOURS
    kwargs['times'] = TIMES
    kwargs['params'] = PARAMS

    # running the algorithm
    ANIM = LabelledTreeAnimation(**kwargs)
    if kwargs['plot_tree']:
        ANIM.tree(name=kwargs['name'])
    if kwargs['animate']:
        ANIM.video(name=kwargs['name'])