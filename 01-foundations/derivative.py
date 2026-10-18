def square(x):
    return x * x

def cube(x):
    return x * 3

def slope(f, x):
    a = f(x)
    b = f(x + 0.001)

    return (b - a) / 0.001

print(slope(square, 10))
