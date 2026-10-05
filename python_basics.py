"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set("aeiouAEIOU")
    return sum(1 for char in text if char in vowels)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count('1')


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    count = 0
    while number >= 10:
        prod = 1
        for digit in str(number):
            prod *= int(digit)
        number = prod
        count += 1
    return count


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    n = len(predicted)
    if n == 0:
        return 0.0
    return sum((p - e) ** 2 for p, e in zip(predicted, expected)) / n


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    if number == 1:
        return ""

    factors = {}
    d = 2
    temp = number
    while d * d <= temp:
        while temp % d == 0:
            factors[d] = factors.get(d, 0) + 1
            temp //= d
        d += 1
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1

    res = []
    for p in sorted(factors.keys()):
        exp = factors[p]
        if exp == 1:
            res.append(f"({p})")
        else:
            res.append(f"({p}**{exp})")

    return "".join(res)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0
    while total < cube_count:
        k += 1
        total += k * k
        if total == cube_count:
            return k
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    s = str(data.value)
    length = len(s)
    half = length // 2

    if length % 2 == 1:
        left = s[:half]
        right = s[half + 1:]
    else:
        left = s[:half - 1]
        right = s[half + 1:]

    sum_left = sum(int(c) for c in left)
    sum_right = sum(int(c) for c in right)

    return sum_left == sum_right
