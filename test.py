from memory_profiler import memory_usage
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a


mem = max(memory_usage(my_func))

print(mem)
