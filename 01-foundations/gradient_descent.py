def loss(w):
    return (w - 5) * (w - 5)

def slope(loss, x):
    a = loss(x)
    b = loss(x + 0.001)
    return (b - a) / 0.001
w = 0
for _ in range(30):
    w = w - (0.1 * slope(loss, w))
    print(w)
