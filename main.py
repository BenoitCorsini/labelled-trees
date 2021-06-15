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

    kwargs = vars(parser.parse_args())

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