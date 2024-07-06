import numpy as np
import matplotlib.pyplot as plt

def perfect_shuffle(arr):
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

def find_shuffle_cycle(m):
    original = list(range(m))
    shuffled = original[:]
    count = 0
    while True:
        shuffled = perfect_shuffle(shuffled)
        count += 1
        if shuffled == original:
            return count

def plot_exponential_fit(max_power):
    powers_of_two = [2**i for i in range(1, max_power + 1)]
    cycle_lengths = [find_shuffle_cycle(m) for m in powers_of_two]

    plt.figure(figsize=(10, 6))
    plt.scatter(powers_of_two, cycle_lengths, label='实践法 (实际值)', color='b', marker='o')

    coefficients = np.polyfit(np.log2(powers_of_two), cycle_lengths, 1)
    polynomial = np.poly1d(coefficients)

    x_fit = np.linspace(min(powers_of_two), max(powers_of_two), 1000)
    y_fit = polynomial(np.log2(x_fit))

    plt.plot(x_fit, y_fit, label='拟合曲线 $log_2(m)$', color='r', linestyle='--')
    plt.xlabel('牌堆大小 (m)', fontsize=14)
    plt.ylabel('恢复原状的次数 (n)', fontsize=14)
    plt.title('牌堆大小与恢复原状次数的关系', fontsize=16)
    plt.legend()
    plt.show()

    y_pred = polynomial(np.log2(powers_of_two))
    r2 = np.corrcoef(cycle_lengths, y_pred)[0, 1]**2
    mse = np.mean((cycle_lengths - y_pred)**2)

    print(f"拟合方程: y = {coefficients[0]:.4f} * log2(x) + {coefficients[1]:.4f}")
    print(f"R²评分: {r2:.4f}")
    print(f"均方误差: {mse:.4f}")

if __name__ == "__main__":
    max_power = 1000  # Example maximum power of 2
    plot_exponential_fit(max_power)
