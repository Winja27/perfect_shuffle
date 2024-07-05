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

def visualize_combined_fit(max_m):
    m_values = list(range(2, max_m + 1))
    cycle_lengths = [find_shuffle_cycle(m) for m in m_values]

    # Extract powers of 2
    powers_of_two = [m for m in m_values if (m & (m - 1)) == 0]
    powers_of_two_indices = [m_values.index(m) for m in powers_of_two]
    powers_of_two_cycle_lengths = [cycle_lengths[i] for i in powers_of_two_indices]

    # Calculate log2(m) for powers of 2
    log2_fitted_cycle_lengths = [np.log2(m) for m in powers_of_two]

    # Fit the remaining data using RandomForestRegressor
    non_powers_of_two_indices = [i for i in range(len(m_values)) if m_values[i] not in powers_of_two]
    non_powers_of_two_m_values = np.array([m_values[i] for i in non_powers_of_two_indices]).reshape(-1, 1)
    non_powers_of_two_cycle_lengths = np.array([cycle_lengths[i] for i in non_powers_of_two_indices])

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(non_powers_of_two_m_values, non_powers_of_two_cycle_lengths)
    rf_fitted_cycle_lengths = rf_model.predict(non_powers_of_two_m_values).round().astype(int)

    # Combine results
    combined_fitted_cycle_lengths = cycle_lengths[:]
    for i, idx in enumerate(powers_of_two_indices):
        combined_fitted_cycle_lengths[idx] = log2_fitted_cycle_lengths[i]
    for i, idx in enumerate(non_powers_of_two_indices):
        combined_fitted_cycle_lengths[idx] = rf_fitted_cycle_lengths[i]

    # Plot original data
    plt.figure(figsize=(10, 6))
    plt.plot(m_values, cycle_lengths, 'o', label='Original Data')

    # Plot combined fitted data
    plt.plot(m_values, combined_fitted_cycle_lengths, 'x', label='Combined Fitted Model')

    plt.title('Cycle Lengths for Perfect Shuffles with Combined Fitting')
    plt.xlabel('Number of Cards (m)')
    plt.ylabel('Cycle Length (n)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    max_m = 1000  # Example maximum value for m
    visualize_combined_fit(max_m)
