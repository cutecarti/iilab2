import math
import sys
from collections import namedtuple

from typing import List
import numpy as np
from sklearn.cluster import KMeans


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

def find_optimal_k_elbow_method(cities: List[City], max_k: int = 10) -> int:
    """
    Метод локтя для нахождения оптимального количества кластеров.
    Возвращает k, при котором наблюдается наибольшее уменьшение инерции.
    """
    if len(cities) <= 2:
        return len(cities)

    inertias = []
    k_range = range(1, max_k + 1)

    for k in k_range:
        clusters = make_clusters_scikitlearn(k,cities)
        # Считаем инерцию (сумму квадратов расстояний до центроидов)
        inertia = 0.0
        for cluster in clusters:
            if not cluster:
                continue
            cx = sum(cities[i].x for i in cluster) / len(cluster)
            cy = sum(cities[i].y for i in cluster) / len(cluster)
            for i in cluster:
                inertia += (cities[i].x - cx) ** 2 + (cities[i].y - cy) ** 2
        inertias.append(inertia)

    # Находим "локоть" — точку с максимальным изменением скорости убывания
    if len(inertias) <= 2:
        return 1

    # Метод: максимальное расстояние от точки до прямой между первой и последней точками
    x1, y1 = 1, inertias[0]
    x2, y2 = max_k, inertias[-1]

    max_dist = -1
    optimal_k = 2

    for i in range(1, len(inertias) - 1):
        xi, yi = i + 1, inertias[i]
        # Расстояние от точки до прямой (x1,y1)-(x2,y2)
        num = abs((y2 - y1) * xi - (x2 - x1) * yi + x2 * y1 - y2 * x1)
        den = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        if den == 0:
            continue
        dist = num / den
        if dist > max_dist:
            max_dist = dist
            optimal_k = i + 1

    print(f"Оптимальное количество кластеров (метод локтя): {optimal_k}")
    return optimal_k

def make_clusters_scikitlearn(k, cities):
    
    # Преобразуем города в numpy массив координат
    coordinates = np.array([[city.x, city.y] for city in cities])
    
    # Применяем KMeans к координатам
    kmeans = KMeans(n_clusters=k, random_state=58, n_init=10)
    labels = kmeans.fit_predict(coordinates)
    
    # Формирование кластеров на основе меток
    clusters = [[] for _ in range(k)]
    for city_idx, label in enumerate(labels):
        clusters[label].append(city_idx)
    
    # Сортируем каждый кластер
    for cluster in clusters:
        cluster.sort()
    
    return clusters

def print_clusters(clusters, cities):
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

    try:
        cities = read_cities(filepath)
    except FileNotFoundError:
        print(f"Error: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    if not cities:
        print("Error: no cities found in the file.", file=sys.stderr)
        sys.exit(1)

    print("\n=== ВЫПОЛНЕНИЕ АЛГОРИТМА КЛАСТЕРИЗАЦИИ ===")

    # Находим оптимальное k с помощью метода локтя
    optimal_k = find_optimal_k_elbow_method(cities)

    clusters = make_clusters_scikitlearn(optimal_k, cities)
    # Выводим кластеры в консоль
    print("\n=== РЕЗУЛЬТИРУЮЩИЕ КЛАСТЕРЫ ===")
    print_clusters(clusters, cities)

if __name__ == "__main__":
    main()