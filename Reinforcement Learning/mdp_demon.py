"""
MDP(Markov decision processes) DEMON
Author: helinfei
"""

# 结论：策略迭代和值迭代结果相同。策略迭代迭代次数更少，要远远快于值迭代。
# 但策略迭代要通过逆矩阵求解线性方程组，如果矩阵位数增加，计算量将急剧上升。
import numpy as np
import time

# initial 初始化五元组：状态state、动作action、转移概率p、阻尼系数gamma、回报函数reward
state = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [1, 2], [3, 2], [1, 3], [2, 3], [3, 3], [4, 2], [4, 3]])
action = {'L': (-1, 0), 'R': (1, 0), 'U': (0, 1), 'D': (0, -1)}
gamma = 0.99
reward = -0.02 * np.ones([11])
reward[9] = -1
reward[10] = 1


def get_p(s1, a, s2):
    """
    s1 经过动作 a 到达 s2 的概率
    :param s1: state 1
    :param a: action s1 -> s2
    :param s2: state 2
    :return: probability
    """
    point1 = state[s1, :]   # s1时候的位置
    point2 = state[s2, :]   # s2时候的位置
    addition = 0            # 四面中有几面有墙壁就加上addition的概率
    flag = 0                # s2是否在s1的四面
    for key in action:
        temp = point1 + action[key]
        if temp.tolist() not in state.tolist():     # 判断是够跑出状态范围
            addition += 0.1
        if temp.tolist() == point2.tolist():
            flag = 1
            if key != a:
                return 0.1
    if not flag:
        return 0
    if point2.tolist() == (point1 + action[a]).tolist():
        return 0.7 + addition
    elif (point1 + action[a]).tolist() not in state.tolist():
        return 0
    else:
        return 0.1


# value iteration
pi = ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L']
V = np.zeros([state.shape[0]])
V[9] = reward[9]
V[10] = reward[10]
err = 1e-15
count_vi = 0    # 值迭代次数
vi_start = time.clock()
while 1:
    count_vi += 1
    Vpre = V.copy()
    for s in range(9):
        temp_vmax = -10000
        for key in action:
            temp = 0
            for s1 in range(11):
                temp += get_p(s, key, s1) * Vpre[s1]
            if reward[s] + gamma * temp >= temp_vmax:
                temp_vmax = reward[s] + gamma * temp
                pi[s] = key         # 保留最优策略
        V[s] = temp_vmax            # 保留最优值函数值
    if np.abs(np.sum(Vpre - V)) <= err:
        break
vi_end = time.clock()
print('值迭代，每一步的值函数值为', V)
print('值迭代，最优决策为', pi)
print('值迭代，迭代%d次' % count_vi)
print('值迭代，耗时%s秒' % (vi_end - vi_start))


# policy iteration
pi = ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L']
V = np.zeros([state.shape[0]])
V[9] = reward[9]
V[10] = reward[10]
err = 1e-15
pa = np.zeros([11, 11])
count_pi = 0
pi_start = time.clock()
while 1:
    pi_pre = pi.copy()
    count_pi += 1
    for i in range(11):
        for j in range(11):
            if i < 9:                       # 终止状态之前的矩阵赋值为I-gamma*pij
                if i == j:
                    pa[i][j] = 1
                else:
                    pa[i][j] = - gamma * get_p(i, pi[i], j)
            else:                           # 终止状态时的矩阵赋值为I
                if i == j:
                    pa[i][j] = 1
                else:
                    pa[i][j] = 0
    V = np.dot(np.linalg.inv(pa), reward)   # 通过逆矩阵来求解线性方程组（bellman公式）
    for s in range(9):
        temp_pimax = -1000
        for key in action:
            temp = 0
            for s1 in range(11):
                temp += get_p(s, key, s1) * V[s1]
            if temp >= temp_pimax:
                temp_pimax = temp
                pi[s] = key     # 找到最优策略
    if pi_pre == pi:
        break
pi_end = time.clock()
print('策略迭代，每一步的值函数值为', V)
print('策略迭代，最优决策为', pi)
print('策略迭代，迭代%d次' % count_pi)
print('策略迭代，耗时%s秒' % (pi_end - pi_start))
