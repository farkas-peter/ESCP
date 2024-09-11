import numpy as np

DYNAMIC_CONSTANTS = {'Izz': 5, 't_w': 0.15, 'm_w': 0.01, 'r': 0.2, 'm_b': 5.0, 'w': 0.5, 'd': 0.,
                     'R': 0.05, 'L': 0.01, 'K1': 1.0, 'K2': 1.0}

DYNAMIC_CONSTANTS_RANGE = {'Izz': (0.35, 10),  # Inertia of the robot. (has high effect on possible trajectory).
                           't_w': 0.15,  # thickness of the wheels (has low effect on trajectories)
                           'm_w': (0.006, 0.5),  # mass of the wheels (has a slight effect on trajectories)
                           'r': 0.2,  # radius of the wheels (has high effect on trajectories)
                           'm_b': (0.23, 5),  # mass of the robot (has low effect on trajectories)
                           'w': (0.16, 0.5),  # width of the robot (has some effect on trajectories)
                           'd': 0.,  # motor distance from center of gravity (has slight effect on trajectories)
                           'R': (0.02, 0.2),  # resistance of the motor (has slight effect on trajectories)
                           'L': (0.005, 0.05),  # inductance of the motor (has slight effect on trajectories)
                           'K1': (0.8, 1.2),  # motor constant (has high effect on trajectories)
                           'K2': (0.8, 1.2)  # motor constant (has high effect on trajectories) should be similar to K1
                           }

HARD_DYNAMIC_CONSTANTS_RANGE = {'Izz': (10.0, 200.0), 't_w': (0.1, 0.3), 'm_w': (0.01, 10.0), 'r': (0.05, 0.5),
                                'm_b': (1.0, 50.0), 'w': (0.1, 0.5), 'd': 0., 'R': (0.005, 0.2), 'L': (0.0001,  0.1),
                                'K1': (0.2, 1.2), 'K2': (0.2, 1.2)}
# todo: don't we want to use the normal dynamic constants range here?
DYNAMIC_CONSTANTS_NORM_FACTS = [max(np.abs(dyn_value)) if isinstance(dyn_value, tuple) else abs(dyn_value) for
                                dyn_value in HARD_DYNAMIC_CONSTANTS_RANGE.values()]

KIN_ACTION_DICT = {0: (0.5, 0.1666 * 5),    # turning right
                   1: (1.0, 0),             # going forward
                   2: (0.5, -0.1666 * 5),   # turning left
                   3: (-0.5, 0.1666 * 5),   # turning right backward
                   4: (-1.0, 0),            # going backward
                   5: (-0.5, -0.1666 * 5),  # turning left backward
                   6: (0., 0.),             # doing nothing
                   7: (0, 0.1666 * 5),      # turning right in one place
                   8: (0, -0.1666 * 5),     # turning left in one place
                   }

DYN_ACTION_DICT = {0: (1., 0.),    # turning right
                   1: (1., 1.),    # going forward
                   2: (0., 1.),    # turning left
                   3: (-1., 0.),   # turning right backward
                   4: (-1., -1.),  # going backward
                   5: (0., -1.),   # turning left backward
                   6: (0., 0.),    # doing nothing
                   7: (1., -1.),   # turning right in one place
                   8: (-1., 1.),   # turning left in one place
                   }
