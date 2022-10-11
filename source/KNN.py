import collections
import math
from typing import Callable

import pandas

def KNN(points: pandas.DataFrame, labels: int, k: int, unknown_point: pandas.DataFrame, metric: Callable[[list[float], list[float]], list[float]]) -> int:
    distances2D = [[metric(unknown, point) for _, point in points.iterrows()] for _, unknown in unknown_point.iterrows()]
    sorted_labels2D = [[label for _, label in sorted(zip(distances, labels))] for distances in distances2D ]

    return [collections.Counter(sorted_labels[:k]).most_common(1)[0][0] for sorted_labels in sorted_labels2D]

def euclidean_distance(a: list[float], b: list[float]):
    diff = [k-l for k, l in zip(a, b)]
    return math.hypot(*diff)
