import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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

def visualize_polynomial_regression(max_m, degree):
    m_values = list(range(2, max_m + 1))
    cycle_lengths = [find_shuffle_cycle(m) for m in m_values]

    # Fit the data using Polynomial Regression
    poly = PolynomialFeatures(degree=degree)
    m_values_reshaped = np.array(m_values).reshape(-1, 1)
    m_poly = poly.fit_transform(m_values_reshaped)
    model = LinearRegression()
    model.fit(m_poly, cycle_lengths)
    fitted_cycle_lengths = model.predict(m_poly)

    # Plot original data
    plt.figure(figsize=(10, 6))
    plt.plot(m_values, cycle_lengths, 'o', label='Data')

    # Plot fitted curve
    plt.plot(m_values, fitted_cycle_lengths, '-', label=f'Polynomial Regression (degree={degree})')
    plt.title('Cycle Lengths for Perfect Shuffles with Polynomial Regression')
    plt.xlabel('Number of Cards (m)')
    plt.ylabel('Cycle Length (n)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    max_m = 1000  # Example maximum value for m
    degree = 4  # Degree of the polynomial
    visualize_polynomial_regression(max_m, degree)
