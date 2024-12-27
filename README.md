# 基于贪心遗传算法求解校内快递配送路径规划

该项目旨在通过改进的贪心遗传算法解决校园内的快递配送路径规划问题，以提高配送效率和降低成本。

## 项目背景

随着网络技术的飞速进步与移动设备的广泛渗透，外卖送餐服务在数字时代背景下迎来了新的快速增长。在高校云集之地，校园内学生群体的集中与较高的消费能力使其自然演变为了第三方快递服务的重要配送区域。在这样的背景下，快递员在执行配送任务时，频繁面临着一个挑战：那就是对于散布不一的大学校园内，如何巧妙规划不同快递点的配送路线。

## 算法介绍

本项目提出了一种改进的贪心遗传算法，该算法通过贪心策略选择距离较近的配送点进行连接，减少了无效的随机选择，提升了种群质量；同时通过贪心交叉和贪心变异策略优先选择距离最短的路径，减少了迭代中的误差和计算复杂度，确保个体进化朝向更优解。

## 实验结果

### 改进遗传算法与传统遗传算法对比

使用贪心遗传算法、传统遗传算法和贪心算法进行了多次独立实验，实验设置为种群规模M=200，最大迭代次数1000代。各算法实验结果对比如下表所示。

| 指标/算法    | 贪心     | 传统遗传 | 贪心遗传 |
| ------------ | -------- | -------- | -------- |
| 运行时间 (s) | 0        | 4.03     | 3.83     |
| 最短距离 (m) | 1604.814 | 1415.188 | 1419.013 |

### 城市规模与各算法关系

由于TSP问题随着城市规模N的增大，总路径长度增大为N!，使用准确求解法的时间复杂度将呈指数级增加，因此我们仅探讨了N=4,8,12,20N=4, 8, 12, 20情况下的时间花销。城市间的距离设定在0到1000之间，遗传算法的其他参数保持不变。实验结果对比如下表所示。

| 指标/算法    | 暴力求解 | 传统遗传 | 贪心遗传 |
| ------------ | -------- | -------- | -------- |
| 运行时间 (s) | 0        | 0.7      | 1.67     |
| 最短距离 (m) | 4.000    | 4.000    | 4.000    |
| **N=4**      |          |          |          |

| 指标/算法    | 暴力求解 | 传统遗传 | 贪心遗传 |
| ------------ | -------- | -------- | -------- |
| 运行时间 (s) | 0.47     | 1.15     | 1.72     |
| 最短距离 (m) | 10.240   | 10.243   | 10.242   |
| **N=8**      |          |          |          |

| 指标/算法    | 暴力求解 | 传统遗传 | 贪心遗传 |
| ------------ | -------- | -------- | -------- |
| 运行时间 (s) | 14958.14 | 1.47     | 2.12     |
| 最短距离 (m) | 16.490   | 16.715   | 16.498   |
| **N=12**     |          |          |          |

| 指标/算法    | 暴力求解 | 传统遗传 | 贪心遗传 |
| ------------ | -------- | -------- | -------- |
| 运行时间 (s) | /        | 2.29     | 2.17     |
| 最短距离 (m) | /        | 29.659   | 29.429   |
| **N=20**     |          |          |          |

## 如何使用
### 环境准备
确保您的环境中已安装以下Python库：
- numpy
- random
- matplotlib
- time

### 运行步骤

1. **导入必要的库**
   ```python
   import numpy as np
   import random
   import matplotlib.pyplot as plt
   import time
   import Greed as gd

2. **定义城市坐标数据**
   城市坐标数据已在代码中预定义，您也可以通过`generate_random_city_map`函数生成随机城市坐标。

3. **设置参数**
   设置种群大小、交叉概率、变异概率和迭代次数等参数。
   ```python
   DNA_SIZE = len(City_Map)
   POP_SIZE = 300
   CROSS_RATE = 0.6
   MUTA_RATE = 0.2
   Iterations = 1000
   ```

4. **计算距离矩阵**
   使用`compute_distance_matrix`函数计算城市之间的距离矩阵。
   ```python
   distance_matrix = compute_distance_matrix(City_Map)
   ```

5. **初始化种群**
   使用贪心算法生成初始种群，然后随机打乱城市顺序生成其他个体。
   ```python
   pop = []
   ll = gd.Inital_pop(City_Map)
   pop.append(ll)
   map_list = list(range(DNA_SIZE))
   for i in range(POP_SIZE - 1):
       random.shuffle(map_list)
       l = map_list.copy()
       pop.append(l)
   ```

6. **迭代优化**
   进行指定次数的迭代，包括交叉、变异、选择和适应度计算。
   ```python
   for i in range(Iterations):
       pop = crossmuta(pop, CROSS_RATE)
       fitness = getfitness(pop, distance_matrix)
       maxfitness = np.argmax(fitness)
       current_best_distance = distance(pop[maxfitness], distance_matrix)
       if current_best_distance < global_best_distance:
           global_best_distance = current_best_distance
           best_iteration = i
       best_dis.append(current_best_distance)
       pop = select(pop, fitness)
   ```

7. **输出结果**
   使用`print_info`函数打印最终结果，包括最优路径和路径图。
   ```python
   print_info(pop, distance_matrix)
   ```

8. **绘制优化过程图**
   绘制并保存路径优化过程图。
   ```python
   plt.plot(range(Iterations), best_dis)
   plt.savefig(r'result\贪心遗传算法路径优化过程图.png', dpi=300)
   ```

## 注意事项
- 确保`Greed`模块正确导入，并且`Inital_pop`函数可用。
- 运行代码前，检查所有参数设置是否符合您的需求。
- 结果将保存在`result`文件夹中。
## 代码结构

- 项目根目录
  │
  ├── result/                # 存放算法执行结果的文件夹
  │
  ├── Greed.py               # 贪心算法函数的Python脚本，用于调用
  │
  ├── 基本遗传算法.py       # 实现基本遗传算法的Python脚本
  │
  ├── 暴力求解算法.py       # 实现暴力求解算法的Python脚本
  │
  ├── 贪心算法.py           # 实现贪心算法的Python脚本
  │
  ├── 贪心遗传_初始值.py   # 实现贪心遗传算法初始化部分的Python脚本
  │
  └── 贪心遗传算法.py       # 实现贪心遗传算法的Python脚本，主算法文件

## 贡献

如果您对项目有任何改进建议或发现问题，欢迎提交Issue或Pull Request。

## 许可证

本项目采用[MIT License](LICENSE)。

## 联系

如有任何问题，请联系项目维护者：[Memory555](2641339226@qq.com)。
