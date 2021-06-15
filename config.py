import numpy as np


COLOURS = {
    'node': 'darkgreen',
    'repeating_node' : 'mediumturquoise',
    'un_node': 'paleturquoise',
    'root': 'limegreen',
    'repeating_root' : 'limegreen',
    'un_root': 'mediumturquoise',
    'leaf': 'crimson',
    'un_leaf': 'paleturquoise',
    'explore': 'gold',
    'change' : 'lightcyan',
    'edge': 'saddlebrown',
    'leaf_edge': 'chocolate',
    'un_edge': 'paleturquoise',
}

TIMES = {
    'start' : 2,
    'appeared' : .5,
    'moment' : 1.5,
    'pause' : 1,
    'break' : .5,
    'explore' : .3,
    'change' : .15,
    'clean' : .1,
    'end' : 3,
}

PARAMS = {
    'shade' : {
        'angle' : -np.pi/6,
        'shift' : 2.7,
        'colour' : 'sandybrown',
        'alpha' : 0.5,
        'n' : 5,
        'start' : 0.5,
    },
    'frame' : {
        'min' : 2,
        'max' : 8,
        'border' : 1.01,
        'colour' : 'wheat',
        'border_colour' : 'peru',
        'alpha_canvas' : 0.8,
        'black_shade' : 0.1,
        'white_shade' : 0.2,
    },
    'zorder' : {
        'shade' : -1,
        'un_covered' : 0,
        'moving' : 1,
        'fixed' : 2,
    },
    'animation' : {
        'extra_edge' : 0.5,
        'distance_to_node' : 1.7,
        'edge_ratio' : 1/3,
        'n_appear' : 5,
        'n_disappear' : 40,
        'quick_growth' : 1.02,
        'n_quick_growth' : 2,
        'n_steps' : 40,
        'step_speed' : 10,
        'n_change' : 3,
        'n_clean' : 2,
        'magnet' : 10,
        'magnet_steps' : 3,
        'n_fade' : 20,
        'fps' : 20,
    },
    'other' : {
        'figure_size' : 10,
        'ratio_point_to_inch' : 750,
        'plot_numbers' : True,
    },
}