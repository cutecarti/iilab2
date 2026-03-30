import math
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


def assign_to_clusters(centers, distances):
    num_cities = len(distances)
    assignments = [-1] * num_cities
    
    for city_idx in range(num_cities):
        closest_center = min(range(len(centers)),
                             key=lambda x: distances[city_idx][centers[x]])
        assignments[city_idx] = closest_center
    
    return assignments


def calculate_average_weight(cluster, center, distances):
    total_weight = sum(distances[center][c] for c in cluster)
    return total_weight / len(cluster)


def find_next_center(centers, remaining_cities, distances):
    min_avg_weight = float('inf')
    next_center = None
    
    for candidate_city in remaining_cities:
        avg_weights = []
        for existing_center in centers:
            cluster = [candidate_city] + \
                     [c for c in remaining_cities if c != candidate_city and
                      distances[candidate_city][existing_center] <= distances[c][existing_center]]
            avg_weight = calculate_average_weight(cluster, existing_center, distances)
            avg_weights.append(avg_weight)
        
        current_min_avg = min(avg_weights)
        if current_min_avg < min_avg_weight:
            min_avg_weight = current_min_avg
            next_center = candidate_city
    
    return next_center


def make_clusters_basic(matrix, k):

    dists = matrix
    num_cities = len(dists)
    
    max_distance = 0
    initial_centers = []
    for i in range(num_cities):
        for j in range(i+1, num_cities):
            if dists[i][j] > max_distance:
                max_distance = dists[i][j]
                initial_centers = [i, j]
    
    centers = initial_centers[:]
    
    # Все города кроме центральных
    remaining_cities = set(range(num_cities)) - set(initial_centers)
    
    # Первые два кластера
    clusters = [[] for _ in range(k)]
    clusters[0].append(initial_centers[0])
    clusters[1].append(initial_centers[1])
    
    # Если k равно количеству городов, значит каждый город сам образует кластер
    if k >= num_cities:
        for idx in range(num_cities):
            clusters[idx].append(idx)
        return clusters
    
    # Создание нужных кластеров
    while len(centers) < k:
        next_center = find_next_center(centers, remaining_cities, dists)
        centers.append(next_center)
        remaining_cities.remove(next_center)
        clusters[len(centers)-1].append(next_center)
    

    assignments = assign_to_clusters(centers, dists)
    for city_idx in remaining_cities:
        assigned_cluster = assignments[city_idx]
        clusters[assigned_cluster].append(city_idx)

    for cluster in clusters:
        cluster.sort()
    
    return clusters

def validate_k(k, num_cities):
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > num_cities:
        raise ValueError(f"k ({k}) exceeds number of cities ({num_cities})")
    

def print_clusters(clusters, cities):
    """
    Печатает кластеры в требуемом формате.
    :param clusters: Список кластеров (каждый кластер - список индексов городов)
    :param cities: Список городов
    """
    result = []
    for cluster in clusters:
        if cluster:  # Проверяем, что кластер не пустой
            city_names = [cities[idx].name for idx in cluster]
            result.append(city_names)
    print(result)


def main():
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
    clusters = make_clusters_basic(matrix, k)
    
    # Выводим кластеры в консоль
    print("\n=== РЕЗУЛЬТИРУЮЩИЕ КЛАСТЕРЫ ===")
    print_clusters(clusters, cities)


if __name__ == '__main__':
    main()