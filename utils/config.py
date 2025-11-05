
# constant Parameters
class const:
    FPS = 30
    CONTROL_RADIUS = 4/2
    BALL_RADIUS = 0.215/2
    WINDOW_SIZE = 55
# static paths
class paths:
    smpl_file = "smpl_models/smpl/SMPL_MALE.pkl"

# joint set
class joint_set:
    leaf = [7, 8, 12, 20, 21]
    full = list(range(0, 24))
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]

    full_parent = [None, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19, 20, 21]

    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)

class style_set:
    demo_style_set = ["beat",
        "celebrate",
        "dribble",
        # "fancy",
        "move",
        "shoot",
        "stand"]
    
