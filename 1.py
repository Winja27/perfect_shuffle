import matplotlib.pyplot as plt

class Solution:
    def reverse(self, a, start, end):
        while start < end:
            a[start], a[end] = a[end], a[start]
            start += 1
            end -= 1

    def left_rotate(self, a, m, n):
        self.reverse(a, 0, m - 1)
        self.reverse(a, m, n - 1)
        self.reverse(a, 0, n - 1)

    def cycle_leader(self, a, start, n):
        pre = a[start]
        mod = 2 * n  # 修改这里
        next_pos = (start * 2) % mod
        while next_pos != start:
            a[next_pos], pre = pre, a[next_pos]
            next_pos = (next_pos * 2) % mod
        a[start] = pre

    def perfect_shuffle(self, a, n):
        while n >= 1:
            k = 0
            r = 3
            while r - 1 <= 2 * n:
                r *= 3
                k += 1
            m = (r // 3 - 1) // 2
            self.left_rotate(a, m, n)
            for i in range(k):
                start = 3 ** i - 1
                self.cycle_leader(a, start, m)
            a = a[2 * m:]  # 修改这里，创建一个新的列表
            n -= m

    def find_shuffling_period(self, m):
        original = list(range(m))
        shuffled = original[:]
        self.perfect_shuffle(shuffled, m // 2)
        period = 1
        while shuffled != original:
            self.perfect_shuffle(shuffled, m // 2)
            period += 1
        return period

# Example usage:
solution = Solution()
periods = []
m_values = []
for m in range(10, 101):
    period = solution.find_shuffling_period(m)
    print(f"The period for {m} cards is: {period}")
    periods.append(period)
    m_values.append(m)

plt.plot(m_values, periods)
plt.xlabel('m')
plt.ylabel('Period')
plt.title('Period of Perfect Shuffle for Different m Values')
plt.grid(True)
plt.show()
