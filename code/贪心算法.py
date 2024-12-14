import numpy as np
import matplotlib.pyplot as plt
import random
import time


# 计算两城市之间的欧几里得距离
def euclidean_distance(city1, city2):
    return np.sqrt(np.sum((city1 - city2) ** 2))


# 贪心算法求解TSP
def greedy_tsp(cities):
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

    return path, total_distance


# 绘制城市路径图
def plot_path(cities, path):
    # 提取城市的坐标
    coords = cities[path]

    # 绘制城市
    plt.figure(1)

    # 绘制路径
    for i in range(len(path) - 1):
        city1, city2 = path[i], path[i + 1]
        plt.plot([cities[city1, 0], cities[city2, 0]], [cities[city1, 1], cities[city2, 1]], color='blue', marker='o', linestyle='-', markersize=5)

    # 返回起点形成闭环
    plt.plot([cities[path[-1], 0], cities[path[0], 0]], [cities[path[-1], 1], cities[path[0], 1]], color='blue', marker='o', linestyle='-', markersize=5)

    # 标注城市坐标
    for i, city in enumerate(cities):
        plt.text(city[0], city[1], str(i), fontsize=9, color='red')

    plt.xlabel('横坐标')
    plt.ylabel('纵坐标')
    plt.title('轨迹图')
    plt.savefig(r'result\贪心算法轨迹图.png', dpi=300)

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

# 主程序入口
if __name__ == "__main__":
    # 设置支持中文的字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    # # City_Map: 城市坐标
    # City_Map = np.array([[951, 205], [891, 153], [875, 220], [874, 145], [915, 243], [764, 28],
    #                      [998, 265], [987, 379], [916, 231], [834, 123], [968, 237], [754, 15],
    #                      [984, 246], [941, 216], [993, 254], [994, 222], [912, 272], [928, 213],
    #                      [931, 200], [649, 104], [948, 214], [903, 153], [943, 207], [916, 154],
    #                      [950, 207], [874, 135], [934, 115], [960, 221], [962, 233], [977, 378],
    #                      [960, 226], [903, 370]])

    # 设置城市数量
    num_cities = 30  # 你可以根据需要修改这个值
    # 生成随机城市坐标
    City_Map = generate_random_city_map(num_cities)

    # 求解TSP问题
    start_time = time.time()  # 记录起始时间
    path, total_distance = greedy_tsp(City_Map)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f"运行时间：{elapsed_time:.2f} 秒")

    # 输出结果
    print("城市访问顺序:", path)
    print("总距离:", total_distance)

    # 绘制路径图
    plot_path(City_Map, path)
