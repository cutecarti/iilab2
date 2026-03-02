"""City clustering using Dijkstra-based shortest-path distances."""

import math
import heapq
import sys
from collections import namedtuple

City = namedtuple('City', ['name', 'x', 'y'])


def parse_city_line(line):
    """Parse a single line into a City. Returns None for blank/malformed lines."""
    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split()
    if len(parts) != 3:
        print(f"Warning: skipping malformed line: {line.rstrip()}", file=sys.stderr)
        return None
    name = parts[0]
    try:
        x = float(parts[1])
        y = float(parts[2])
    except ValueError:
        print(f"Warning: skipping malformed line: {line.rstrip()}", file=sys.stderr)
        return None
    return City(name, x, y)


def read_cities(filepath):
    """Read cities from a file. Raises FileNotFoundError if file is missing."""
    cities = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            city = parse_city_line(line)
            if city is not None:
                cities.append(city)
    return cities


def euclidean_distance(c1, c2):
    """Euclidean distance between two City objects."""
    return math.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)


def build_adjacency_matrix(cities):
    """Build NxN symmetric adjacency matrix of Euclidean distances. Diagonal = 0."""
    n = len(cities)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(cities[i], cities[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


def dijkstra(matrix, source):
    """Dijkstra's shortest path from source to all nodes. Returns list of distances."""
    n = len(matrix)
    dist = [math.inf] * n
    dist[source] = 0.0
    visited = [False] * n
    heap = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        for v in range(n):
            if visited[v]:
                continue
            weight = matrix[u][v]
            if weight > 0 and d + weight < dist[v]:
                dist[v] = d + weight
                heapq.heappush(heap, (dist[v], v))

    return dist


def select_seeds(matrix, k):
    """Select k seed indices via farthest-first traversal.

    Returns (seeds, seed_dists) where seed_dists[i] is the Dijkstra
    distance array from seeds[i] to all nodes.
    """
    n = len(matrix)
    seeds = [0]
    seed_dists = [dijkstra(matrix, 0)]

    # min_dist[j] = min distance from any current seed to node j
    min_dist = list(seed_dists[0])

    for _ in range(1, k):
        # pick the node farthest from all current seeds
        farthest = -1
        farthest_dist = -1.0
        for j in range(n):
            if j not in seeds and min_dist[j] > farthest_dist:
                farthest_dist = min_dist[j]
                farthest = j
        seeds.append(farthest)
        new_dists = dijkstra(matrix, farthest)
        seed_dists.append(new_dists)
        # update min distances
        for j in range(n):
            if new_dists[j] < min_dist[j]:
                min_dist[j] = new_dists[j]

    return seeds, seed_dists


def assign_clusters(seed_dists, seeds, cities):
    """Assign each city to the nearest seed. Returns dict: seed_index -> [city_names]."""
    n = len(cities)
    clusters = {seed: [] for seed in seeds}
    for j in range(n):
        best_seed = seeds[0]
        best_dist = seed_dists[0][j]
        for i, seed in enumerate(seeds):
            if seed_dists[i][j] < best_dist:
                best_dist = seed_dists[i][j]
                best_seed = seed
        clusters[best_seed].append(cities[j].name)
    return clusters


def validate_k(k, num_cities):
    """Validate k. Raises ValueError if invalid."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > num_cities:
        raise ValueError(f"k ({k}) exceeds number of cities ({num_cities})")


def print_clusters(clusters, seeds, cities):
    """Pretty-print clusters."""
    for i, seed in enumerate(seeds):
        seed_name = cities[seed].name
        members = clusters[seed]
        print(f"Cluster {i + 1} (seed: {seed_name}): {', '.join(members)}")


def main():
    """Main entry point: read cities, cluster, print results."""
    if len(sys.argv) < 2:
        filepath = "cities.txt"
    else:
        filepath = sys.argv[1]

    if len(sys.argv) >= 3:
        k_str = sys.argv[2]
    else:
        k_str = "3"

    try:
        k = int(k_str)
    except ValueError:
        print(f"Error: k must be an integer, got '{k_str}'", file=sys.stderr)
        sys.exit(1)

    try:
        cities = read_cities(filepath)
    except FileNotFoundError:
        print(f"Error: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    if not cities:
        print("Error: no cities found in the file.", file=sys.stderr)
        sys.exit(1)

    try:
        validate_k(k, len(cities))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    matrix = build_adjacency_matrix(cities)
    seeds, seed_dists = select_seeds(matrix, k)
    clusters = assign_clusters(seed_dists, seeds, cities)
    print_clusters(clusters, seeds, cities)


if __name__ == '__main__':
    main()
