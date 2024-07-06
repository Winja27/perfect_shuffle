import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 假设我们已经有deck_sizes和cycles的数据
deck_sizes = np.arange(2, 520)  # 2到51张牌
cycles = []


def perfect_shuffle_practice(arr):
    n = len(arr)
    if n < 3:
        return arr

    b = [0] * n
    mid = (n + 1) // 2
    for i in range(mid):
        b[2 * i] = arr[i]
    for i in range(mid, n):
        b[2 * (i - mid) + 1] = arr[i]
    return b


def find_shuffle_cycle_practice(m):
    original = list(range(m))
    shuffled = original[:]
    count = 0
    while True:
        shuffled = perfect_shuffle_practice(shuffled)
        count += 1
        if shuffled == original:
            return count


# 计算cycles
for size in deck_sizes:
    cycle = find_shuffle_cycle_practice(size)
    cycles.append(cycle)

# 转换为numpy数组
deck_sizes = np.array(deck_sizes)
cycles = np.array(cycles)

# 数据可视化
plt.figure(figsize=(10, 6))
plt.scatter(deck_sizes, cycles, label='实践法 (实际值)', color='b', marker='o')
plt.xlabel('牌堆大小', fontsize=14)
plt.ylabel('恢复原状的次数', fontsize=14)
plt.title('牌堆大小与恢复原状次数的关系', fontsize=16)
plt.legend()
plt.show()


# 分段线性拟合
def piecewise_linear_fit(x, y, breakpoints):
    models = []
    for i in range(len(breakpoints) - 1):
        mask = (x >= breakpoints[i]) & (x < breakpoints[i + 1])
        x_segment = x[mask].reshape(-1, 1)
        y_segment = y[mask]

        model = LinearRegression()
        model.fit(x_segment, y_segment)
        models.append((model, breakpoints[i], breakpoints[i + 1]))
    return models


# 假设我们根据数据分布选择了一些断点
breakpoints = [2, 10, 20, 30, 40, 52]

models = piecewise_linear_fit(deck_sizes, cycles, breakpoints)

plt.figure(figsize=(10, 6))
plt.scatter(deck_sizes, cycles, label='实践法 (实际值)', color='b', marker='o')

for model, start, end in models:
    x_fit = np.linspace(start, end, 100).reshape(-1, 1)
    y_fit = model.predict(x_fit)
    plt.plot(x_fit, y_fit, label=f'线性拟合 [{start}, {end})')

plt.xlabel('牌堆大小', fontsize=14)
plt.ylabel('恢复原状的次数', fontsize=14)
plt.title('牌堆大小与恢复原状次数的关系', fontsize=16)
plt.legend()
plt.show()

for model, start, end in models:
    print(f"分段 [{start}, {end}): 斜率 = {model.coef_[0]}, 截距 = {model.intercept_}")
