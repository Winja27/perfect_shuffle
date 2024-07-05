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

def visualize_shuffle_cycles(max_m):
    m_values = list(range(2, max_m + 1))
    cycle_lengths = [find_shuffle_cycle(m) for m in m_values]

    plt.figure(figsize=(10, 6))
    plt.plot(m_values, cycle_lengths, marker='o')
    plt.title('Cycle Lengths for Perfect Shuffles')
    plt.xlabel('Number of Cards (m)')
    plt.ylabel('Cycle Length')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    max_m = 100  # Example maximum value for m
    visualize_shuffle_cycles(max_m)
