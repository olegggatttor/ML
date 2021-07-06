from math import sqrt

n = int(input())
xs = []
ys = []

E_x = 0
E_y = 0

for _ in range(n):
    x, y = list(map(float, input().split()))
    xs.append(x)
    ys.append(y)
    E_x += x
    E_y += y

E_x /= n
E_y /= n

kek = sqrt(sum([(x - E_x) ** 2 for x in xs]) * sum([(y - E_y) ** 2 for y in ys]))

if kek == 0:
    print(0)
else:
    print(sum([(x - E_x) * (y - E_y) for x, y in zip(xs, ys)]) /kek)
