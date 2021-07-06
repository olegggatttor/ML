n = int(input())

xs = []
ys = []

numToPosX = dict()
numToPosY = dict()

for i in range(n):
    x, y = list(map(float, input().split()))
    numToPosX.setdefault(x, i)
    numToPosY.setdefault(y, i)
    xs.append(x)
    ys.append(y)


xs.sort()
ys.sort()

resX = [0] * n
resY = [0] * n

for i in range(n):
    resX[numToPosX[xs[i]]] = i
    resY[numToPosY[ys[i]]] = i

print(format(1.0 - 6.0 * sum(map(lambda x: (x[0] - x[1]) ** 2, zip(resX, resY)))/(n ** 3 - n), ".9f"))