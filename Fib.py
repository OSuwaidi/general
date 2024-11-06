def fib(num):
    if num <= 1:
        return num

    ans_o = 0
    ans_n = 1
    for i in range(num-1):
        temp = ans_n
        ans_n += ans_o
        ans_o = temp
    return ans_n


print(fib(7))
