def series(j, n):
    # Fractional part of: sum over k of 16^(n-k) / (8k+j)
    total = 0.0

    # Part 1: k = 0..n, use modular power to keep numbers small
    for k in range(n + 1):
        denominator = 8 * k + j
        total = (total + pow(16, n - k, denominator) / denominator) % 1

    # Part 2: k > n, terms shrink fast, add until they stop mattering
    k = n + 1
    while True:
        term = 16 ** (n - k) / (8 * k + j)
        if term < 1e-17:
            break
        total += term
        k += 1

    return total % 1


def calculateDigit(n):
    # n = 1 returns the leading 3; n = 2 onward returns hex digits after the point
    if n == 1:
        return "3"
    position = n - 2
    x = (4 * series(1, position)
         - 2 * series(4, position)
         - series(5, position)
         - series(6, position)) % 1
    return "%X" % int(x * 16)


userinput = int(input("which digit of pi do you want to calculate? : "))
print(f'the {userinput}th hex digit of pi is {calculateDigit(userinput)}')