def f():
    x = 1

    def g():
        nonlocal x

        x += 1

        print(x)

    return g


a = f()

b = f()

a()

b()

a()

b()