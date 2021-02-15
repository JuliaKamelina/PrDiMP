from .get_sequence_info import get_sequence_info
from .load_video_info import load_video_info, load_video_info_test
from .tensorlist import TensorList
from .plotting import *
from .tracking import *

def _round(x):
    res = x.copy()
    res[0] = np.ceil(x[0]) if x[0] - np.floor(x[0]) >= 0.5 else np.floor(x[0])
    res[1] = np.ceil(x[1]) if x[1] - np.floor(x[1]) >= 0.5 else np.floor(x[1])
    return res
