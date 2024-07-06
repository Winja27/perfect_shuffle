import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor
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

# 将数据转换为二维数组用于聚类
data = np.column_stack((deck_sizes, cycles))

# 使用RANSAC进行拟合
ransac = RANSACRegressor(LinearRegression(), residual_threshold=50)
ransac.fit(data[:, 0].reshape(-1, 1), data[:, 1])

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

x_inliers = data[inlier_mask][:, 0].reshape(-1, 1)
y_inliers = data[inlier_mask][:, 1]

def piecewise_linear(x, slopes, intercepts, breakpoints):
    y = np.zeros_like(x)
    for i in range(len(slopes)):
        mask = (x >= breakpoints[i]) & (x < breakpoints[i+1])
        y[mask] = slopes[i] * x[mask] + intercepts[i]
    return y

def fit_segments(x, y, segments):
    model = LinearRegression()
    slopes = []
    intercepts = []
    breakpoints = np.linspace(x.min(), x.max(), segments + 1)
    x = x.flatten()
    for i in range(segments):
        mask = (x >= breakpoints[i]) & (x < breakpoints[i + 1])
        x_segment = x[mask]
        y_segment = y[mask]
        if len(x_segment) > 0:
            model.fit(x_segment.reshape(-1, 1), y_segment)
            slopes.append(model.coef_[0])
            intercepts.append(model.intercept_)
    return np.array(slopes), np.array(intercepts), breakpoints

segments = 3  # 设定划分的区段数
slopes, intercepts, breakpoints = fit_segments(x_inliers, y_inliers, segments)

plt.figure(figsize=(10, 6))
plt.scatter(deck_sizes, cycles, label='实践法 (实际值)', color='b', marker='o')
plt.scatter(x_inliers, y_inliers, label='内点 (拟合数据)', color='r', marker='x')

x_segment = np.linspace(x_inliers.min(), x_inliers.max(), 100).reshape(-1, 1)
y_segment = piecewise_linear(x_segment, slopes, intercepts, breakpoints)
plt.plot(x_segment, y_segment, label='分段线性拟合', color='g')

plt.xlabel('牌堆大小', fontsize=14)
plt.ylabel('恢复原状的次数', fontsize=14)
plt.title('牌堆大小与恢复原状次数的关系（分段线性拟合）', fontsize=16)
plt.legend()
plt.show()

y_pred = piecewise_linear(x_inliers, slopes, intercepts, breakpoints)
r2 = r2_score(y_inliers, y_pred)
mse = mean_squared_error(y_inliers, y_pred)

print(f"R²评分: {r2}")
print(f"均方误差: {mse}")

def predict_cycle(deck_size):
    if x_inliers.min() <= deck_size <= x_inliers.max():
        segment = np.searchsorted(breakpoints, deck_size) - 1
        return slopes[segment] * deck_size + intercepts[segment]
    else:
        return None

test_deck_sizes = [8, 12, 18, 22]
for size in test_deck_sizes:
    cycle = predict_cycle(size)
    if cycle is not None:
        print(f"Deck size: {size}, Predicted cycle: {cycle}")
    else:
        print(f"Deck size: {size}, Not using the linear model")
