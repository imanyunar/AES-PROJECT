import warnings
warnings.filterwarnings("ignore")

import numpy as np
import json

# =====================================================
# HAMMING WEIGHT LOOKUP (NumPy 2.x safe & fast)
# =====================================================
HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

# =====================================================
# CONSTANT
# =====================================================
C = np.array([1,1,0,0,0,1,1,0], dtype=np.uint8)

# =====================================================
# MULTIPLICATIVE INVERSE TABLE
# =====================================================
INV = np.array([
[0,1,141,246,203,82,123,209,232,79,41,192,176,225,229,199],
[116,180,170,75,153,43,96,95,88,63,253,204,255,64,238,178],
[58,110,90,241,85,77,168,201,193,10,152,21,48,68,162,194],
[44,69,146,108,243,57,102,66,242,53,32,111,119,187,89,25],
[29,254,55,103,45,49,245,105,167,100,171,19,84,37,233,9],
[237,92,5,202,76,36,135,191,24,62,34,240,81,236,97,23],
[22,94,175,211,73,166,54,67,244,71,145,223,51,147,33,59],
[121,183,151,133,16,181,186,60,182,112,208,6,161,250,129,130],
[131,126,127,128,150,115,190,86,155,158,149,217,247,2,185,164],
[222,106,50,109,216,138,132,114,42,20,159,136,249,220,137,154],
[251,124,46,195,143,184,101,72,38,200,18,74,206,231,210,98],
[12,224,31,239,17,117,120,113,165,142,118,61,189,188,134,87],
[11,40,47,163,218,212,228,15,169,39,83,4,27,252,172,230],
[122,7,174,99,197,219,226,234,148,139,196,213,157,248,144,107],
[177,13,214,235,198,14,207,173,8,78,215,227,93,80,30,179],
[91,35,56,52,104,70,3,140,221,156,125,160,205,26,65,28]
], dtype=np.uint8)

# =====================================================
# AFFINE MATRICES (PERSIS DARI KAMU)
# =====================================================
A0 = np.array([
[1,1,0,0,0,0,0,1],
[1,1,1,0,0,0,0,0],
[0,1,1,1,0,0,0,0],
[0,0,1,1,1,0,0,0],
[0,0,0,1,1,1,0,0],
[0,0,0,0,1,1,1,0],
[0,0,0,0,0,1,1,1],
[1,0,0,0,0,0,1,1]
], dtype=np.uint8)

A1 = np.array([
[0,1,1,1,1,0,1,0],
[0,0,1,1,1,1,0,1],
[1,0,0,1,1,1,1,0],
[0,1,0,0,1,1,1,1],
[1,0,1,0,0,1,1,1],
[1,1,0,1,0,0,1,1],
[1,1,1,0,1,0,0,1],
[1,1,1,1,0,1,0,0]
], dtype=np.uint8)

A2 = np.array([
[1,1,1,0,1,0,0,1],
[1,1,1,1,0,1,0,0],
[0,1,1,1,1,0,1,0],
[0,0,1,1,1,1,0,1],
[1,0,0,1,1,1,1,0],
[0,1,0,0,1,1,1,1],
[1,0,1,0,0,1,1,1],
[1,1,0,1,0,0,1,1]
], dtype=np.uint8)

K4 = np.array([
[0,0,0,0,0,1,1,1],
[1,0,0,0,0,0,1,1],
[1,1,0,0,0,0,0,1],
[1,1,1,0,0,0,0,0],
[0,1,1,1,0,0,0,0],
[0,0,1,1,1,0,0,0],
[0,0,0,1,1,1,0,0],
[0,0,0,0,1,1,1,1]
], dtype=np.uint8)

K44 = np.array([
[0,1,0,1,0,1,1,1],
[1,0,1,0,1,0,1,1],
[1,1,0,1,0,1,0,1],
[1,1,1,0,1,0,1,0],
[0,1,1,1,0,1,0,1],
[1,0,1,1,1,0,1,0],
[0,1,0,1,1,1,0,1],
[1,0,1,0,1,1,1,0]
], dtype=np.uint8)

K128 = np.array([
[1,1,1,1,1,1,1,0],
[0,1,1,1,1,1,1,1],
[1,0,1,1,1,1,1,1],
[1,1,0,1,1,1,1,1],
[1,1,1,0,1,1,1,1],
[1,1,1,1,0,1,1,1],
[1,1,1,1,1,0,1,1],
[1,1,1,1,1,1,0,1]
], dtype=np.uint8)

# =====================================================
# CONSTRUCT S-BOX
# =====================================================
def construct_sbox(A):
    s = np.zeros(256, dtype=np.uint8)
    for x in range(256):
        inv = INV[x >> 4, x & 0xF]
        bits = (inv >> np.arange(8)) & 1
        out = (A @ bits + C) & 1
        s[x] = np.sum(out << np.arange(8))
    return s

# =====================================================
# GENERATE ALL S-BOXES
# =====================================================
def generate_sboxes(include_random=True, seed=None):
    sboxes = {
        "A0": construct_sbox(A0),
        "A1": construct_sbox(A1),
        "A2": construct_sbox(A2),
        "K4": construct_sbox(K4),
        "K44": construct_sbox(K44),
        "K128": construct_sbox(K128)
    }

    AES = np.array([
        0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
        0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
        0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
        0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
        0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
        0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
        0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
        0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
        0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
        0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
        0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
        0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
        0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
        0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
        0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
        0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
    ], dtype=np.uint8)

    sboxes["AES"] = AES

    if include_random:
        if seed is not None:
            np.random.seed(seed)
        sboxes["RANDOM"] = np.random.permutation(256).astype(np.uint8)

    return sboxes

# =====================================================
# FWHT
# =====================================================
def fwht(a):
    a = a.astype(np.int32)
    h = 1
    while h < len(a):
        for i in range(0, len(a), h*2):
            x = a[i:i+h]
            y = a[i+h:i+2*h]
            a[i:i+h] = x + y
            a[i+h:i+2*h] = x - y
        h <<= 1
    return a

# =====================================================
# METRICS
# =====================================================
def balance(s):
    return all(np.sum((s >> b) & 1) == 128 for b in range(8))

def bijective(s):
    return np.unique(s).size == 256

def nonlinearity(s):
    bits = ((s[:,None] >> np.arange(8)) & 1)
    nl = 256
    for b in range(8):
        f = 1 - 2*bits[:,b]
        W = np.abs(fwht(f))
        nl = min(nl, 128 - np.max(W)//2)
    return int(nl)

def sac(s):
    idx = np.arange(256)
    total = 0
    for i in range(8):
        total += np.sum(HW[s ^ s[idx ^ (1<<i)]])
    return total / (256*8*8)

def lap(s):
    max_bias = 0
    x = np.arange(256)
    for a in range(1,256):
        pa = HW[x & a] & 1
        for b in range(1,256):
            pb = HW[s & b] & 1
            cnt = np.sum(pa == pb)
            max_bias = max(max_bias, abs(cnt-128)/256)
    return max_bias**2

def dap(s):
    maxv = 0
    for dx in range(1,256):
        dy = s ^ s[np.arange(256) ^ dx]
        maxv = max(maxv, np.max(np.bincount(dy)))
    return maxv / 256

def bic_sac_fast(s):
    idx = np.arange(256)
    total = count = 0
    for i in range(8):
        di = s ^ s[idx ^ (1<<i)]
        bi = (di[:,None] >> np.arange(8)) & 1
        for j in range(i+1,8):
            dj = s ^ s[idx ^ (1<<j)]
            bj = (dj[:,None] >> np.arange(8)) & 1
            total += np.sum(bi ^ bj)
            count += bi.size
    return total / count

def bic_nl_fast(s):
    bits = ((s[:,None] >> np.arange(8)) & 1)
    max_bias = 0
    for i in range(8):
        for j in range(i+1,8):
            f = 1 - 2*(bits[:,i] ^ bits[:,j])
            W = np.abs(fwht(f))
            max_bias = max(max_bias, np.max(W)//2)
    return 128 - max_bias

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    sboxes = generate_sboxes(include_random=True, seed=42)
    results = []

    for name, s in sboxes.items():
        res = {
            "S-box": name,
            "Balance": balance(s),
            "Bijective": bijective(s),
            "NL": nonlinearity(s),
            "SAC": sac(s),
            "LAP": lap(s),
            "DAP": dap(s),
            "BIC-SAC": bic_sac_fast(s),
            "BIC-NL": float(bic_nl_fast(s))
        }

        results.append(res)

        print(f"\n===== HASIL UJI S-BOX {name} =====")
        for k, v in res.items():
            print(f"{k:10}: {v}")

    with open("all_sbox_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n✅ Semua hasil disimpan ke all_sbox_results.json")

