import numpy as np

def euclidean_distance(city1, city2):
    return np.sqrt(np.sum((city1 - city2) ** 2))

def Inital_pop(cities):
    # 初始化变量
    num_cities = len(cities)
    visited = [False] * num_cities  # 记录是否访问过某个城市
    path = []  # 存储访问的城市路径
    total_distance = 0  # 总距离

    # 从第一个城市开始
    current_city = 0
    visited[current_city] = True
    path.append(current_city)

    # 访问剩余的城市
    for _ in range(num_cities - 1):
        min_distance = float('inf')
        next_city = -1

        # 找到距离当前城市最近的未访问城市
        for i in range(num_cities):
            if not visited[i]:
                distance = euclidean_distance(cities[current_city], cities[i])
                if distance < min_distance:
                    min_distance = distance
                    next_city = i

        # 更新状态
        visited[next_city] = True
        path.append(next_city)
        total_distance += min_distance
        current_city = next_city

    # 最后返回起点城市，形成一个闭环
    total_distance += euclidean_distance(cities[current_city], cities[path[0]])

    return path