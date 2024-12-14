import numpy as np
import matplotlib.pyplot as plt
import random

# 处理数据
coord = np.array([[951,205],[891,153],[875,220],[874,145],[915,243],[764,28],
         [998,265],[987,379],[916,231],[834,123],[968,237],[754,15],
         [984,246],[941,216],[993,254],[994,222],[912,272],[928,213],
         [931,200],[649,104],[948,214],[903,153],[943,207],[916,154],
         [950,207],[874,135],[934,115],[960,221],[962,233],[977,378],
         [960,226],[903,370]])

w, h = coord.shape
coordinates = np.zeros((w, h), float)
for i in range(w):
    for j in range(h):
        coordinates[i, j] = float(coord[i, j])
# print(coordinates)

# 距离矩阵
distance = np.zeros((w, w))
for i in range(w):
    for j in range(w):
        distance[i, j] = distance[j, i] = np.linalg.norm(coordinates[i] - coordinates[j]) # 欧氏距离

# 种群数量
count = 200
# 进化次数
iter_time = 1000
# 最优选择概率
retain_rate = 0.3 # 适应度前30%存活
# 弱者生存概率
random_select_rate = 0.5
# 变异
mutation_rate = 0.1
# 改良初始化种群
# 为了初始化一个较好的种群，如果随即交换两个城市的位置，如果总距离减小，那么就更新这个染色体
gailiang_N = 3000

# 适应度
def get_total_distance(x):# x为一条染色体
    dista = 0
    for i in range(len(x)):
        if i == len(x) - 1:
            dista += distance[x[i]][x[0]]
        else:
            dista += distance[x[i]][x[i+1]]
    return dista

# 初始种群的改良
def gailiang(x):
    distance = get_total_distance(x)
    gailiang_num = 0
    while gailiang_num < gailiang_N:
        while True:
            a = random.randint(0, len(x) - 1)
            b = random.randint(0, len(x) - 1)
            if a != b:
                break
        new_x = x.copy()
        temp_a = new_x[a]
        new_x[a] = new_x[b]
        new_x[b] = temp_a
        if get_total_distance(new_x) < distance:
            x = new_x.copy()
        gailiang_num += 1

# 自然选择
def nature_select(population):
    grad = [[x, get_total_distance(x)] for x in population]
    grad = [x[0] for x in sorted(grad, key=lambda x:x[1])] # 按照距离排序
    # 强者
    retain_length = int(retain_rate * len(grad)) # 距离短的部分遗传
    parents = grad[: retain_length] # 适应度前30%存活
    # 生存下来的弱者
    for weaker in grad[retain_length: ]: # 适应度后70%的个体
        if random.random() < random_select_rate: # 弱者生存概率：0.5
            parents.append(weaker)
    return parents

# 交叉繁殖
def crossover(parents):
    target_count = count - len(parents)# 种群数量-父代数量
    children = []
    while len(children) < target_count:
        while True:  # 选择父代种群中不同的两个染色体分别做父亲和母亲
            male_index = random.randint(0, len(parents)-1)
            female_index = random.randint(0, len(parents)-1)
            if male_index != female_index:
                break
        male = parents[male_index]
        female = parents[female_index]
        left = random.randint(0, len(male)-2)
        right = random.randint(left, len(male)-1)
        # 从父亲和母亲染色体中分别取出基因片段
        gen_male = male[left:right]
        gen_female = female[left:right]
        child_a = []
        child_b = []

        len_ca = 0
        for g in male:
            if len_ca == left:
                child_a.extend(gen_female)
                len_ca += len(gen_female)
            if g not in gen_female:
                child_a.append(g)
                len_ca += 1
        len_cb = 0
        for g in female:
            if len_cb == left:
                child_b.extend(gen_male)
                len_cb += len(gen_male)
            if g not in gen_male:
                child_b.append(g)
                len_cb += 1
        children.append(child_a)
        children.append(child_b)
    return children

# 变异
def mutation(children):
    for i in range(len(children)):
        if random.random() < mutation_rate:
            while True:
                u = random.randint(0, len(children[i])-1)
                v = random.randint(0, len(children[i])-1)
                if u != v:
                    break
            temp_a = children[i][u]
            children[i][u] = children[i][v]
            children[i][v] = temp_a

def get_result(population):
    grad = [[x, get_total_distance(x)]for x in population]
    grad = sorted(grad, key=lambda x:x[1])
    return grad[0][0], grad[0][1]

population = []
# 初始化种群
index = [i for i in range(w)]
for i in range(count):
    x = index.copy()
    random.shuffle(x)
    # gailiang(x)
    population.append(x)

distance_list = []
result_cur_best, dist_cur_best = get_result(population)
distance_list.append(dist_cur_best)

i = 0
while i < iter_time:
    # 自然选择
    parents = nature_select(population)
    # 繁殖
    children = crossover(parents)
    # 变异
    mutation(children)
    # 更新
    population = parents + children  # 遗传的优良基因+遗传的部分弱者+用遗传的父代交叉产生的孩子

    result_cur_best, dist_cur_best = get_result(population)
    distance_list.append(dist_cur_best)
    i = i+1
    print(result_cur_best)
    print(dist_cur_best)

for i in range(len(result_cur_best)):
    result_cur_best[i] += 1

result_path = result_cur_best
result_path.append(result_path[0])
print(result_path)

# 画图

X = []
Y = []
for index in result_path:
    X.append(coordinates[index-1, 0])
    Y.append(coordinates[index-1, 1])

plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False
plt.figure(1)
plt.plot(X, Y, '-o')
for i in range(len(X)):
    plt.text(X[i] + 0.05, Y[i] + 0.05, str(result_path[i]), color='red')
plt.xlabel('横坐标')
plt.ylabel('纵坐标')
plt.title('轨迹图')

plt.figure(2)
plt.plot(np.array(distance_list))
plt.title('优化过程')
plt.ylabel('最优值')
plt.xlabel('代数({}->{})'.format(0, iter_time))
plt.show()