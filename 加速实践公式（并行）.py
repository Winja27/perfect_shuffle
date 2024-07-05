import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from math import gcd

# 实践法
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

# 计算欧拉函数 φ(n)
def euler_totient(n):
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result

# 公式法
def shuffle_times_to_restore_formula(deck_size):
    if deck_size % 2 == 0:
        mod = deck_size - 1
    else:
        mod = deck_size

    # 计算欧拉函数 φ(mod)
    phi = euler_totient(mod)

    # 找到最小的 t 使得 2^t ≡ 1 (mod)
    t = 1
    power_of_two = 2
    while power_of_two % mod != 1:
        power_of_two = (power_of_two * 2) % mod
        t += 1

    return t

# 并行计算
deck_sizes = list(range(2, 52))  # 2到51张牌

with ThreadPoolExecutor() as executor:
    practice_values = list(executor.map(find_shuffle_cycle_practice, deck_sizes))
    formula_values = list(executor.map(shuffle_times_to_restore_formula, deck_sizes))

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(deck_sizes, practice_values, label='实践法 (实际值)', marker='o')
plt.plot(deck_sizes, formula_values, label='公式法 (解析值)', marker='x')
plt.xlabel('牌堆大小')
plt.ylabel('恢复原状的次数')
plt.title('实践法与公式法计算的恢复原状次数对比')
plt.legend()
plt.grid(True)
plt.show()
