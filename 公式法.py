import matplotlib.pyplot as plt
import numpy as np

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

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

def shuffle_period(n):
    if n % 2 == 0:
        return 2 * euler_totient(n // 2)
    else:
        return euler_totient(n)

def calculate_periods(max_n):
    periods = []
    for n in range(1, max_n + 1):
        periods.append(shuffle_period(n))
    return periods

def visualize_periods(periods):
    x = np.arange(1, len(periods) + 1)
    y = np.array(periods)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='b', marker='o')
    plt.title('Shuffle Periods for Different n')
    plt.xlabel('n')
    plt.ylabel('Shuffle Period')
    plt.grid(True)
    plt.show()

# Define the maximum value of n
max_n = 1000

# Calculate periods
periods = calculate_periods(max_n)

# Visualize the results
visualize_periods(periods)
