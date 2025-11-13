import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS

def plot_routes_from_distance_matrix(distance_matrix, routes, objective=-1, depot_index=0, title="OR Tools", total_distance=None):
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)

    sym_matrix = (np.array(distance_matrix) + np.array(distance_matrix).T) / 2
    coordinates = mds.fit_transform(sym_matrix)

    plt.figure(figsize=(12, 9))

    for route in routes:
        if not route:
            continue
        full_route = [depot_index] + route + [depot_index]
        x = [coordinates[i][0] for i in full_route]
        y = [coordinates[i][1] for i in full_route]
        plt.plot(x, y, marker='o', linewidth=1.5)

    depot_x, depot_y = coordinates[depot_index]
    plt.scatter(depot_x, depot_y, c='red', marker='*', s=300, label="Depot")

    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.title(f"{title}\nTotal distance: {objective:.1f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()