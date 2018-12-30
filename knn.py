"""
K-nearest neighbors algo.

blah.
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import Counter


# Make a point object; a little clearer this way.
class point:
    def __init__(self, coords, label):
        self.label = label
        self.coords = coords
        self.distance = None

# Make some fake data
arr = [point([9, 1], 'r'),
       point([9, 1], 'r'),
       point([9, 1], 'r'),
       point([9, 1], 'r'),
       point([1, 9], 'b'),
       point([1, 9], 'b'),
       point([1, 9], 'b'),
       point([1, 9], 'b'),
       point([1, 9], 'b')]



def knn_classifier(new_point, arr, k=5,
                   add_point=True,
                   show_plot=True):
    """
    A brute-force nearest-neighbor classifier.

    Could also do this as a graph matrix, precomputing all the distances.
    That'd be cool.

    Args:
        k (int): number of nearest neighbors to count
        new_point (list): a vector describing the point we're looking at.
                          [coord1, coord2, label]
        arr (list of lists): a list of point objects.
    """
    if k > len(arr):
        return

    def get_distance(p1, p2):
        d = np.sqrt((p1.coords[0] - p2.coords[0])**2 + \
                    (p1.coords[1] - p2.coords[1])**2)

        ndims = len(p1.coords)
        d = np.sqrt(sum([(p1.coords[i] - p2.coords[i])**2 for i in range(ndims)]
                        )
                    )
        return d

    # Calculate all the distances
    new_arr = deepcopy(arr)
    for i, p in enumerate(new_arr):
        new_arr[i].distance = get_distance(p, new_point)

    # Sort by distance, grab the first k
    new_arr.sort(key=lambda x: x.distance)
    nns = new_arr[:k]

    # Extract the labels of those top k, grab the most frequent one.
    lab = max(Counter([el.label for el in nns]))
    new_point.label = lab

    if show_plot is True:
        [plt.plot(el.coords[0], el.coords[1], marker='.', color=el.label) for el in arr]
        plt.plot(new_point.coords[0], new_point.coords[0],
                 marker='o', color=el.label)
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.show()

    if add_point is True:
        arr.append(new_point)

    return lab

knn_classifier(point([5, 5], None), arr)









# The End
