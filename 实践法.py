import matplotlib.pyplot as plt

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

def main():
    deck_sizes = list(range(2, 5200))  # 2到51张牌
    cycles = []

    for size in deck_sizes:
        cycle = find_shuffle_cycle_practice(size)
        cycles.append(cycle)
        print(f"Deck size: {size}, Shuffle cycles to restore: {cycle}")

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(deck_sizes, cycles, label='实践法 (实际值)', color='b', marker='o')
    plt.xlabel('牌堆大小', fontsize=14)
    plt.ylabel('恢复原状的次数', fontsize=14)
    plt.title('牌堆大小与恢复原状次数的关系', fontsize=16)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
