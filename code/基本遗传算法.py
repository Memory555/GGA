import numpy as np
import matplotlib.pyplot as plt
import random
import time

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

# 初始化参数
City_Map = np.array([[951, 205], [891, 153], [875, 220], [874, 145], [915, 243], [764, 28],
                  [998, 265], [987, 379], [916, 231], [834, 123], [968, 237], [754, 15],
                  [984, 246], [941, 216], [993, 254], [994, 222], [912, 272], [928, 213],
                  [931, 200], [649, 104], [948, 214], [903, 153], [943, 207], [916, 154],
                  [950, 207], [874, 135], [934, 115], [960, 221], [962, 233], [977, 378],
                  [960, 226], [903, 370]])

# # 设置城市数量
# num_cities = 12  # 你可以根据需要修改这个值
#
# # 生成随机城市坐标
# City_Map = generate_random_city_map(num_cities)

# City_Map = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # 4个城市的示例

# City_Map = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 2], [3, 2], [3, 3], [2, 3]]) # 城市 8

# 12个城市的坐标
# City_Map = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
#                      [2, 2], [3, 2], [3, 3], [2, 3],
#                      [4, 4], [5, 4], [5, 5], [4, 5]])

# 20个城市的坐标
# City_Map = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
#                      [2, 2], [3, 2], [3, 3], [2, 3],
#                      [4, 4], [5, 4], [5, 5], [4, 5],
#                      [6, 6], [7, 6], [7, 7], [6, 7],
#                      [8, 8], [9, 8], [9, 9], [8, 9]])


w, h = City_Map.shape
coordinates = City_Map.astype(float)  # 转换为浮点数

# 计算距离矩阵
distance = np.zeros((w, w))
for i in range(w):
    for j in range(w):
        distance[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])  # 计算欧几里得距离

# 参数设置
count = 200  # 种群数量
iter_time = 1000  # 迭代次数
retain_rate = 0.3  # 保留的优良个体比例
random_select_rate = 0.5  # 弱者存活概率
mutation_rate = 0.1  # 变异概率
gailiang_N = 3000  # 改良次数

# 适应度函数
def get_total_distance(x):
    dista = 0
    for i in range(len(x)):
        if i == len(x) - 1:
            dista += distance[x[i]][x[0]]  # 最后一段到起点的距离
        else:
            dista += distance[x[i]][x[i + 1]]
    return dista

# 改良函数
def gailiang(x):
    dista = get_total_distance(x)
    gailiang_num = 0
    while gailiang_num < gailiang_N:
        a, b = random.sample(range(len(x)), 2)  # 随机选取两个不同的位置
        new_x = x.copy()
        new_x[a], new_x[b] = new_x[b], new_x[a]  # 交换
        if get_total_distance(new_x) < dista:
            x = new_x.copy()
        gailiang_num += 1

# 自然选择
def nature_select(population):
    graded = sorted(population, key=get_total_distance)  # 按适应度排序
    retain_length = int(retain_rate * len(graded))
    parents = graded[:retain_length]  # 前30%存活
    for weaker in graded[retain_length:]:
        if random.random() < random_select_rate:  # 随机保留部分较差个体
            parents.append(weaker)
    return parents

# 交叉繁殖
def crossover(parents):
    target_count = count - len(parents)
    children = []
    while len(children) < target_count:
        male, female = random.sample(parents, 2)
        left = random.randint(0, len(male) - 2)
        right = random.randint(left, len(male) - 1)
        gen_male = male[left:right]
        child = [g for g in female if g not in gen_male]
        children.append(child[:left] + gen_male + child[left:])
    return children

# 变异
def mutation(children):
    for i in range(len(children)):
        if random.random() < mutation_rate:
            u, v = random.sample(range(len(children[i])), 2)
            children[i][u], children[i][v] = children[i][v], children[i][u]

# 获取当前最优结果
def get_result(population):
    best = min(population, key=get_total_distance)
    return best, get_total_distance(best)

if __name__ == "__main__":
    # 初始化种群
    index = list(range(w))
    population = [random.sample(index, w) for _ in range(count)]

    list = list(range(w))
    for i in range(w - 1):
        random.shuffle(list)  # 打乱城市顺序
        l = list.copy()
        population.append(l)



    # 记录最优值变化
    distance_list = []
    result_cur_best, dist_cur_best = get_result(population)
    distance_list.append(dist_cur_best)

    # 记录最优解首次出现的迭代代数
    best_distance = float('inf')
    first_iteration = None
    best_path = None

    start_time = time.time()  # 记录起始时间
    for i in range(iter_time):
        # 自然选择
        parents = nature_select(population)
        # 交叉繁殖
        children = crossover(parents)
        # 变异
        mutation(children)
        # 更新种群
        population = parents + children

        result_cur_best, dist_cur_best = get_result(population)
        distance_list.append(dist_cur_best)

        if dist_cur_best < best_distance:
            best_distance = dist_cur_best
            first_iteration = i
            best_path = result_cur_best

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f"运行时间：{elapsed_time:.2f} 秒")

    # 输出结果
    print('最优解首次出现的迭代代数：', first_iteration)
    print('最终最优解的距离：', best_distance)
    print('最优的基因型：', best_path)

    result_path = best_path + [best_path[0]]  # 闭环路径

    # 设置支持中文的字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

    # 绘制轨迹图
    X, Y = coordinates[result_path, 0], coordinates[result_path, 1]
    plt.figure(1)
    # plt.plot(X, Y, '-o')
    plt.plot(X, Y, color='blue', marker='o', linestyle='-', markersize=5)
    for i in range(len(X)):
        plt.text(X[i], Y[i], str(result_path[i] + 1), color='red', fontsize=9)
    plt.xlabel('横坐标')
    plt.ylabel('纵坐标')
    plt.title('轨迹图')
    plt.savefig(r'result\基本遗传算法轨迹图.png', dpi=300)
    plt.close()  # 关闭当前图形窗口

    # 绘制优化过程
    plt.figure(2)
    plt.plot(distance_list)
    plt.xlabel('迭代代数(0->1000)')
    plt.ylabel('最优路径长度')
    plt.title('路径优化过程')
    plt.savefig(r'result\基本遗传算法路径优化过程图.png', dpi=300)
    plt.close()  # 关闭当前图形窗口