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

    # Plot original data
    plt.figure(figsize=(10, 6))
    plt.scatter(powers_of_two, cycle_lengths, color='red', label='Data')

    # Fit the data using log2 function
    fitted_cycle_lengths = [np.log2(m) for m in powers_of_two]

    # Plot fitted curve
    plt.plot(powers_of_two, fitted_cycle_lengths, label='Fitted $log_2(m)$', linestyle='--')

    plt.title('Cycle Lengths for Perfect Shuffles (Powers of 2)')
    plt.xlabel('Number of Cards (m)')
    plt.ylabel('Cycle Length (n)')
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

if __name__ == "__main__":
    max_power = 10  # Example maximum power of 2
    plot_exponential_fit(max_power)
