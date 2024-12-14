import numpy as np
import random
import matplotlib.pyplot as plt
import time
import Greed as gd

# 城市坐标数据
# City_Map = np.array([[951,205],[891,153],[875,220],[874,145],[915,243],[764,28],
#          [998,265],[987,379],[916,231],[834,123],[968,237],[754,15],
#          [984,246],[941,216],[993,254],[994,222],[912,272],[928,213],
#          [931,200],[649,104],[948,214],[903,153],[943,207],[916,154],
#          [950,207],[874,135],[934,115],[960,221],[962,233],[977,378],
#          [960,226],[903,370]])

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

# 设置城市数量
num_cities = 30  # 你可以根据需要修改这个值

# 生成随机城市坐标
City_Map = generate_random_city_map(num_cities)

# DNA_SIZE是城市的数量
DNA_SIZE = len(City_Map)
# 种群的大小
POP_SIZE = 300
# 交叉概率
CROSS_RATE = 0.6
# 变异概率
MUTA_RATE = 0.2
# 迭代次数
Iterations = 1000

# 计算城市之间的距离矩阵
def compute_distance_matrix(City_Map):
    num_cities = len(City_Map)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            dist = np.linalg.norm(City_Map[i] - City_Map[j])  # 计算欧几里得距离
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # 距离矩阵是对称的
    return distance_matrix

# 根据DNA编码计算总的旅行距离
def distance(DNA, distance_matrix):
    dis = 0
    for i in range(len(DNA) - 1):
        dis += distance_matrix[DNA[i], DNA[i + 1]]  # 查找距离矩阵中的值
    dis += distance_matrix[DNA[-1], DNA[0]]  # 回到起始城市
    return dis

# 计算种群的适应度，适应度是距离的倒数
def getfitness(pop, distance_matrix):
    temp = []
    for i in range(len(pop)):
        temp.append(1 / (distance(pop[i], distance_matrix)))  # 适应度越大，表示距离越短
    return np.array(temp) - np.min(temp)

# 选择操作，根据适应度选择个体，适应度越大的个体被选中的概率越大
def select(pop, fitness):
    cum_probs = np.cumsum(fitness / np.sum(fitness))
    selected_indices = np.searchsorted(cum_probs, np.random.rand(POP_SIZE))
    return [pop[i] for i in selected_indices]

# 变异操作
def mutation(DNA, MUTA_RATE):
    if np.random.rand() < MUTA_RATE:
        mutate_point1, mutate_point2 = random.sample(range(DNA_SIZE), 2)
        DNA[mutate_point1], DNA[mutate_point2] = DNA[mutate_point2], DNA[mutate_point1]

# 交叉操作
def crossmuta(pop, CROSS_RATE):
    new_pop = []
    for i in range(len(pop)):
        n = np.random.rand()
        if n >= CROSS_RATE:  # 如果随机数大于交叉概率，则直接将父代复制到下一代
            temp = pop[i].copy()
            new_pop.append(temp)

        if n < CROSS_RATE:  # 否则进行交叉操作
            list1 = pop[i].copy()
            list2 = pop[np.random.randint(POP_SIZE)].copy()  # 随机选择另一个个体进行交叉

            status = True
            while status:  # 产生两个不相等的节点，进行部分匹配交叉
                k1 = random.randint(0, len(list1) - 1)
                k2 = random.randint(0, len(list2) - 1)
                if k1 < k2:
                    status = False

            k11 = k1
            fragment1 = list1[k1: k2]
            fragment2 = list2[k1: k2]

            list1[k1: k2] = fragment2
            list2[k1: k2] = fragment1

            del list1[k1: k2]
            left1 = list1

            offspring1 = []
            for pos in left1:
                if pos in fragment2:
                    pos = fragment1[fragment2.index(pos)]
                    while pos in fragment2:
                        pos = fragment1[fragment2.index(pos)]
                    offspring1.append(pos)
                    continue
                offspring1.append(pos)
            for i in range(0, len(fragment2)):
                offspring1.insert(k11, fragment2[i])
                k11 += 1
            temp = offspring1.copy()
            mutation(temp, MUTA_RATE)  # 对子代进行变异

            new_pop.append(temp)  # 将交叉生成的个体加入到新种群中
    return new_pop

# 输出结果信息
# 输出结果信息
def print_info(pop, distance_matrix):
    fitness = getfitness(pop, distance_matrix)
    maxfitness = np.argmax(fitness)  # 获取适应度最强个体的索引
    print("最优的基因型：", pop[maxfitness])  # 输出最优个体
    best_map = [City_Map[i] for i in pop[maxfitness]]
    best_map.append(City_Map[pop[maxfitness][0]])  # 把起始城市加入到最后
    X = np.array(best_map)[:, 0]  # X坐标
    Y = np.array(best_map)[:, 1]  # Y坐标

    # 绘制路径图
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.scatter(X, Y)  # 绘制城市点

    # 用 plt.plot 来连接城市点
    plt.plot(X, Y, color='blue', marker='o', linestyle='-', markersize=5)

    # 在每个点上添加编号
    for dot in range(len(X) - 1):
        plt.annotate(pop[maxfitness][dot], xy=(X[dot], Y[dot]), xytext=(X[dot], Y[dot]))
    plt.annotate('start', xy=(X[0], Y[0]), xytext=(X[0] + 1, Y[0] + 1))
    plt.xlabel("横坐标")
    plt.ylabel("纵坐标")
    plt.title("轨迹图")
    plt.savefig(r'result\贪心遗传算法轨迹图.png', dpi=300)
    plt.close()  # 关闭当前图形窗口



# 主程序
if __name__ == "__main__":
    distance_matrix = compute_distance_matrix(City_Map)  # 预计算距离矩阵

    pop = []  # 初始种群
    ll = gd.Inital_pop(City_Map)  # 使用贪心算法生成初代种群
    pop.append(ll)

    map_list = list(range(DNA_SIZE))
    for i in range(POP_SIZE - 1):
        random.shuffle(map_list)  # 打乱城市顺序
        l = map_list.copy()
        pop.append(l)

    best_dis = []  # 存储每代最优解的距离
    best_iteration = -1  # 记录最优解出现的迭代代数
    global_best_distance = float('inf')  # 初始化全局最优距离为正无穷

    start_time = time.time()  # 记录起始时间
    for i in range(Iterations):  # 迭代指定次数
        pop = crossmuta(pop, CROSS_RATE)  # 进行交叉变异操作
        fitness = getfitness(pop, distance_matrix)
        maxfitness = np.argmax(fitness)
        current_best_distance = distance(pop[maxfitness], distance_matrix)  # 当前最优解的距离

        # 如果发现新的更优解，更新最优解及迭代代数
        if current_best_distance < global_best_distance:
            global_best_distance = current_best_distance
            best_iteration = i  # 更新最优解对应的迭代代数

        best_dis.append(current_best_distance)  # 记录当前最优解的距离
        pop = select(pop, fitness)  # 选择下一代种群

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f"运行时间：{elapsed_time:.2f} 秒")
    print_info(pop, distance_matrix)  # 打印最终结果

    print(f"最优解首次出现的迭代代数: {best_iteration}")
    print(f"最终最优解的距离: {global_best_distance}")

    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 绘制优化过程图
    plt.figure()
    plt.plot(range(Iterations), best_dis)
    plt.xlabel("迭代代数(0->1000)")
    plt.ylabel("最优路径长度")
    plt.title("路径优化过程")
    plt.savefig(r'result\贪心遗传算法路径优化过程图.png', dpi=300)
    plt.close()  # 关闭当前图形窗口

