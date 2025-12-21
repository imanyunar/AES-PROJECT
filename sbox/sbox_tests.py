import numpy as np

def balance(s):
    return all(np.sum((s >> b) & 1) == 128 for b in range(8))

def bijective(s):
    return len(set(s)) == 256

def nonlinearity(s):
    nl = 256
    for a in range(1,256):
        for b in range(1,256):
            cnt = 0
            for x in range(256):
                if (bin(x & a).count("1") ^ bin(s[x] & b).count("1")) % 2 == 0:
                    cnt += 1
            nl = min(nl, 128 - abs(cnt - 128))
    return nl

def sac(s):
    total = 0
    for x in range(256):
        for i in range(8):
            total += bin(s[x] ^ s[x ^ (1 << i)]).count("1")
    return total / (256 * 8 * 8)

def lap(s):
    max_bias = 0
    for a in range(1,256):
        for b in range(1,256):
            cnt = sum(
                (bin(x & a).count("1") ^ bin(s[x] & b).count("1")) % 2 == 0
                for x in range(256)
            )
            max_bias = max(max_bias, abs(cnt - 128) / 256)
    return max_bias ** 2

def dap(s):
    table = np.zeros((256,256), dtype=int)
    for x in range(256):
        for dx in range(1,256):
            dy = s[x] ^ s[x ^ dx]
            table[dx, dy] += 1
    return np.max(table[1:]) / 256
