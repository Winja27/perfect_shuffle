import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

deck_sizes = np.arange(2, 1000)
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

for size in deck_sizes:
    cycle = find_shuffle_cycle_practice(size)
    cycles.append(cycle)

deck_sizes = np.array(deck_sizes)
cycles = np.array(cycles)

plt.figure(figsize=(10, 6))
plt.scatter(deck_sizes, cycles, label='实践法 (实际值)', color='b', marker='o')
plt.xlabel('牌堆大小', fontsize=14)
plt.ylabel('恢复原状的次数', fontsize=14)
plt.title('牌堆大小与恢复原状次数的关系', fontsize=16)
plt.legend()
plt.show()

# 数据建模：尝试多项式拟合
degree = 3  # 假设一个三次多项式
coefficients = np.polyfit(deck_sizes, cycles, degree)
polynomial = np.poly1d(coefficients)

# 计算拟合曲线的y值
x_fit = np.linspace(deck_sizes.min(), deck_sizes.max(), 1000)
y_fit = polynomial(x_fit)

plt.figure(figsize=(10, 6))
plt.scatter(deck_sizes, cycles, label='实践法 (实际值)', color='b', marker='o')
plt.plot(x_fit, y_fit, label=f'{degree}次多项式拟合', color='r')
plt.xlabel('牌堆大小', fontsize=14)
plt.ylabel('恢复原状的次数', fontsize=14)
plt.title('牌堆大小与恢复原状次数的关系', fontsize=16)
plt.legend()
plt.show()

print(f"多项式系数: {coefficients}")

y_pred = polynomial(deck_sizes)
r2 = r2_score(cycles, y_pred)
mse = mean_squared_error(cycles, y_pred)

print(f"R²评分: {r2}")
print(f"均方误差: {mse}")
