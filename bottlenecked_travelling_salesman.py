import numpy as np
import matplotlib.pyplot as plt


def generate_points_and_distances(n, seed=None):
    if (seed is not None):
        np.random.seed(seed)
    points = np.random.rand(n, 2)
    # Compute pairwise distances
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)

    return points, distances

def bottlenecked_travelling_salesman_cutting_edges(n, k, seed=None):
    # Generate random points
    points, distances = generate_points_and_distances(n, seed)

    x_inds, y_inds = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    inds = np.stack((x_inds, y_inds), axis=-1)
    inds = np.reshape(inds, (-1, 2))

    pr = distances.copy()
    pr_bit_mask = np.ones((n, n), dtype=bool)
    pr_bit_mask = pr_bit_mask & np.logical_not(np.eye(n, dtype=bool))
    distances_flat = distances.flatten()

    sorted_distances = np.sort(distances.flatten())[::-1]
    sorted_distances_inds = np.argsort(distances.flatten())[::-1]

    for i in range(0,n**2,2):
        x, y = inds[sorted_distances_inds[i]]

        if(np.sum(pr_bit_mask[x, :]) == 2 or
                np.sum(pr_bit_mask[y,:]) == 2 or
                np.sum(pr_bit_mask[:,x]) == 2 or
                np.sum(pr_bit_mask[:,y]) == 2):
            continue
        if(x == y):
            continue

        #if(np.sum(pr_bit_mask[x,:]) == 3 or
        #   np.sum(pr_bit_mask[y,:]) == 3 or
        #   np.sum(pr_bit_mask[:,x]) == 3 or
        #   np.sum(pr_bit_mask[:,y]) == 3):
        #    #Choose the next best edge
        #    temp_inds = np.where(pr_bit_mask[x,:])

        pr[x, y] = 0
        pr[y, x] = 0
        pr_bit_mask[x, y] = False
        pr_bit_mask[y, x] = False


    journey = np.where(pr_bit_mask)

    journey_dict = dict()
    for i in range(len(journey[0])):
        if journey[0][i] in journey_dict:
            journey_dict[journey[0][i]].append(journey[1][i])
        else:
            journey_dict[journey[0][i]] = [journey[1][i]]

    return pr, journey_dict



if __name__ == '__main__':

    n = 100
    k = 3
    seed = 42069
    points, distances = generate_points_and_distances(n, seed=seed)

    pr, journey = bottlenecked_travelling_salesman_cutting_edges(n, k, seed=seed)

    #Plot the journey
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1])

    for key, val in journey.items():
        for v in val:
            plt.plot([points[key, 0], points[v, 0]], [points[key, 1], points[v, 1]], 'r')

    plt.show()