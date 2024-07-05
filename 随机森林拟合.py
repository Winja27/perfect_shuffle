import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

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

    # Fit the data using RandomForestRegressor
    m_values = np.array(m_values).reshape(-1, 1)
    cycle_lengths = np.array(cycle_lengths)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(m_values, cycle_lengths)
    fitted_cycle_lengths = rf_model.predict(m_values).round().astype(int)

    # Plot fitted discrete data
    plt.plot(m_values, fitted_cycle_lengths, 'x', label='Fitted Random Forest Model')
    plt.title('Cycle Lengths for Perfect Shuffles with Fitted Random Forest Model')
    plt.xlabel('Number of Cards (m)')
    plt.ylabel('Cycle Length (n)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    max_m = 2000  # Example maximum value for m
    visualize_shuffle_cycles(max_m)
