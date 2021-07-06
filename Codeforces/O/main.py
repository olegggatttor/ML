from collections import defaultdict

k = int(input())
n = int(input())

exSq = 0.
pX = defaultdict(float)
pYgivenX = defaultdict(float)

for i in range(n):
    x, y = list(map(float, input().split()))
    exSq += y * (y / n)
    pX[x] += 1.0 / n
    pYgivenX[x] += y / n

print(exSq - sum(map(lambda x: x[0] * (x[0] / x[1]) if x[1] != 0 else 0, zip(pYgivenX.values(), pX.values()))))
