from collections import defaultdict, deque
from itertools import accumulate
from typing import Tuple, List


class DTWAlign(object):
    def __init__(self):
        pass

    def align(self, axis1: List[Tuple[int, int]], axis2: List[Tuple[int, int]]):
        """
        Given two series timestamp, map timestamps on axis2 to that on axis1
        :return aligned timestamp, a deque contains triplet tuple,
                (second, mapped nanosecond on axis1, origin nanosecond of axis2)
        """

        # test case for head and rear padding
        # axis1 = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2),]
        # axis2 = [(2, 1), (2, 2)]
        cache1, cache2 = defaultdict(list), defaultdict(list)
        for s, ns in axis1:
            cache1[s].append(ns)
        for s, ns in axis2:
            cache2[s].append(ns)

        # head and rear padding
        # if axis1 stars before axis2, then extend the head of axis2
        if axis1[0][0] < axis2[0][0]:
            s2, ns2 = axis2[0][0], axis2[0][1]
            for k, v in cache1.items():
                if k < s2:
                    cache2[k] = [ns2, ] * len(v)
        # if axis1 ends after axis2, then extend the rear of axis2
        if axis1[-1][0] > axis2[-1][0]:
            s2, ns2 = axis2[-1][0], axis2[-1][1]
            for k, v in cache1.items():
                if k > s2:
                    cache2[k] = [ns2, ] * len(v)

        align_res = deque()
        res = list(self.dtw(v, cache2[s], s) for s, v in cache1.items())

        for i in res:
            align_res.extend(i)
        return align_res

    @staticmethod
    def dtw(seq1: List, seq2: List, s):
        # test case
        # seq1 = [0, 1, 2, 3, 4, 5]
        # seq2 = [1, 1, 1, 1, 1]

        m, n = len(seq1), len(seq2)

        dis = [[float("inf"), ] * n for i in range(m)]
        for i, v in enumerate(seq1):
            for j, k in enumerate(seq2):
                dis[i][j] = abs(v - k)

        dp = [[float("inf"), ] * n for i in range(m)]
        dp[0] = list(accumulate(dis[0]))
        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] + dis[i][0]
            for j in range(1, n):
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + dis[i][j]

        i, j = m - 1, n - 1
        route = deque()
        while i >= 0 and j >= 0:
            route.appendleft((i, j))
            l = dp[i][j - 1] if j - 1 >= 0 else float("inf")
            u = dp[i - 1][j] if i - 1 >= 0 else float("inf")
            lu = dp[i - 1][j - 1] if i - 1 < m and j - 1 < n else float("inf")
            # encourage it to go the shortest way
            if lu <= u and lu <= l:
                i -= 1
                j -= 1
            elif u <= l:
                i -= 1
            else:
                j -= 1

        # remove duplicates by seq1
        cache = dict()
        for i, j in route:
            cache[i] = j

        return tuple((s, seq1[i], seq2[j]) for i, j in cache.items())


if __name__ == '__main__':
    a = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), ]
    b = [(2, 1), (2, 2)]
    test = DTWAlign()
    c = test.align(a, b)
