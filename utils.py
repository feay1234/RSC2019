import random


def negative_sample(pool, pos):
    neg = random.sample(pool, 1)[0]
    while neg == pos:
        neg = random.sample(pool, 1)[0]
    return neg


