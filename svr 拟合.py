import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

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

def visualize_shuffle_cycles(max_m):
    m_values = list(range(2, max_m + 1))
    cycle_lengths = [find_shuffle_cycle(m) for m in m_values]

    # Plot original data
    plt.figure(figsize=(10, 6))
    plt.plot(m_values, cycle_lengths, 'o', label='Data')

    # Fit the data using SVR
    m_values = np.array(m_values).reshape(-1, 1)
    cycle_lengths = np.array(cycle_lengths)
    svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr_model.fit(m_values, cycle_lengths)
    fitted_cycle_lengths = svr_model.predict(m_values)

    # Plot fitted curve
    plt.plot(m_values, fitted_cycle_lengths, '-', label='Fitted SVR Model')
    plt.title('Cycle Lengths for Perfect Shuffles with Fitted SVR Model')
    plt.xlabel('Number of Cards (m)')
    plt.ylabel('Cycle Length (n)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    max_m = 1000  # Example maximum value for m
    visualize_shuffle_cycles(max_m)
