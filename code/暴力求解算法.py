import numpy as np
import itertools
import matplotlib.pyplot as plt
import random
import time

# 计算两城市间的欧几里得距离
def euclidean_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


# 计算给定路径的总距离
def total_distance(path, city_map):
    distance = 0
    for i in range(len(path) - 1):
        distance += euclidean_distance(city_map[path[i]], city_map[path[i + 1]])
    # 最后从最后一个城市回到起点
    distance += euclidean_distance(city_map[path[-1]], city_map[path[0]])
    return distance


def solve_tsp(city_map):
    # 城市数量
    num_cities = len(city_map)

    # 生成所有城市的排列组合
    city_indices = list(range(num_cities))
    permutations = itertools.permutations(city_indices)

    # 初始化最短距离和最短路径
    min_distance = float('inf')
    best_path = None

    # 暴力求解所有路径
    total_permutations = 0  # 记录排列的总数
    for perm in permutations:
        total_permutations += 1
        dist = total_distance(perm, city_map)
        if dist < min_distance:
            min_distance = dist
            best_path = perm

        # 打印当前进度
        if total_permutations % 1000 == 0:
            print(f"已计算 {total_permutations} 条路径...")

    return best_path, min_distance


def plot_path(city_map, path):
    # 绘制城市
    plt.figure(figsize=(8, 6))

    # 绘制路径
    for i in range(len(path) - 1):
        city1, city2 = path[i], path[i + 1]
        plt.plot([city_map[city1, 0], city_map[city2, 0]], [city_map[city1, 1], city_map[city2, 1]], color='blue',
                 marker='o', linestyle='-', markersize=5)

    # 返回起点形成闭环
    plt.plot([city_map[path[-1], 0], city_map[path[0], 0]], [city_map[path[-1], 1], city_map[path[0], 1]], color='blue',
             marker='o', linestyle='-', markersize=5)

    # 标注城市坐标
    for i, city in enumerate(city_map):
        plt.text(city[0], city[1], str(i), fontsize=9, color='red')

    # 设置图表
    plt.xlabel('横坐标')
    plt.ylabel('纵坐标')
    plt.title('TSP 最短路径图')

    # 保存路径图
    plt.savefig(r'result\暴力求解算法轨迹图.png', dpi=300)
    plt.close()

def generate_random_city_map(num_cities, x_range=(0, 1000), y_range=(0, 1000)):
    """
    生成随机城市坐标
    :param num_cities: 城市数量
    :param x_range: 横坐标的范围，默认 (0, 1000)
    :param y_range: 纵坐标的范围，默认 (0, 1000)
    :return: 返回一个包含城市坐标的 numpy 数组
    """
    city_map = np.array([[random.randint(x_range[0], x_range[1]), random.randint(y_range[0], y_range[1])]
                          for _ in range(num_cities)])
    return city_map

if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    # City_Map: 城市坐标
    # City_Map = np.array([[951, 205], [891, 153], [875, 220], [874, 145], [915, 243], [764, 28],
    #                      [998, 265], [987, 379], [916, 231], [834, 123], [968, 237], [754, 15],
    #                      [984, 246], [941, 216], [993, 254], [994, 222], [912, 272], [928, 213],
    #                      [931, 200], [649, 104], [948, 214], [903, 153], [943, 207], [916, 154],
    #                      [950, 207], [874, 135], [934, 115], [960, 221], [962, 233], [977, 378],
    #                      [960, 226], [903, 370]]) // 32个点太多了

    # City_Map = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 2]])  # 5个城市的示例

    # # 设置城市数量
    # num_cities = 12  # 你可以根据需要修改这个值
    #
    # # 生成随机城市坐标
    # City_Map = generate_random_city_map(num_cities)

    # City_Map = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # 4个城市的示例

    # City_Map = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 2], [3, 2], [3, 3], [2, 3]]) # 城市 8

    # 12个城市的坐标
    City_Map = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
                         [2, 2], [3, 2], [3, 3], [2, 3],
                         [4, 4], [5, 4], [5, 5], [4, 5]])

    # 20个城市的坐标
    # City_Map = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
    #                      [2, 2], [3, 2], [3, 3], [2, 3],
    #                      [4, 4], [5, 4], [5, 5], [4, 5],
    #                      [6, 6], [7, 6], [7, 7], [6, 7],
    #                      [8, 8], [9, 8], [9, 9], [8, 9]])

    # 求解TSP问题
    start_time = time.time()  # 记录起始时间
    best_path, min_distance = solve_tsp(City_Map)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f"运行时间：{elapsed_time:.2f} 秒")

    # 输出最短路径和距离
    print(f"最短路径: {best_path}")
    print(f"最短路径的距离: {min_distance:.2f}")

    # 绘制路径图
    plot_path(City_Map, best_path)
