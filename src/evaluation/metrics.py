import math
import numpy as np

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    if n == 0:
        return (0.0, 0.0, 0.0)
    m = np.mean(a)
    se = np.std(a, ddof=1) / math.sqrt(n) if n > 1 else 0.0
    # t-critical for large n ~ z
    from scipy.stats import t
    h = se * t.ppf((1 + confidence) / 2., n-1) if n > 1 else 0.0
    return m, m - h, m + h
