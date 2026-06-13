import numpy as np
from config import Precision, PRECISION_TO_CPU_DTYPE


def get_dtype(precision: Precision):
    dtype_str = PRECISION_TO_CPU_DTYPE[precision]
    return np.dtype(dtype_str).type


def create_array(shape, precision: Precision):
    dtype = get_dtype(precision)
    rng = np.random.default_rng(42)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return rng.integers(info.min // 2, info.max // 2, shape, dtype=dtype)
    else:
        return rng.random(shape, dtype=dtype)
