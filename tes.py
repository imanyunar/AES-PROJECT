import random
from collections import Counter

AES_POLY = 0x11B  # x^8 + x^4 + x^3 + x + 1

# ========== GF(2^8) Operations ==========
def gf_mul(a, b):
    """Multiply two elements in GF(2^8)"""
    res = 0
    while b:
        if b & 1:
            res ^= a
        a <<= 1
        if a & 0x100:
            a ^= AES_POLY
        b >>= 1
    return res & 0xFF

def gf_pow(a, e):
    """Exponentiation in GF(2^8)"""
    res = 1
    base = a
    while e:
        if e & 1:
            res = gf_mul(res, base)
        base = gf_mul(base, base)
        e >>= 1
    return res

def gf_inv_safe(a):
    """Multiplicative inverse in GF(2^8), returns 0 for 0"""
    return 0 if a == 0 else gf_pow(a, 254)

# ========== Multiplicative Inverse Matrix Generation ==========
def generate_multiplicative_inverse_matrix():
    """Generate the 16x16 multiplicative inverse matrix"""
    matrix = []
    for row in range(16):
        row_values = []
        for col in range(16):
            x = row * 16 + col
            inv = gf_inv_safe(x)
            row_values.append(inv)
        matrix.append(row_values)
    return matrix

# ========== Bit/Byte Conversion ==========
def byte_to_bits(b):
    return [(b >> (7-i)) & 1 for i in range(8)]

def bits_to_byte(bits):
    b = 0
    for i in range(8):
        b = (b << 1) | (bits[i] & 1)
    return b

# ========== Matrix Operations over GF(2) ==========
def matrix_vec_mul(M, v):
    out = [0] * 8
    for i in range(8):
        s = 0
        for j in range(8):
            s ^= (M[i][j] & v[j])
        out[i] = s & 1
    return out

def matrix_is_invertible(M):
    A = [row[:] for row in M]
    n = 8
    rank = 0
    for col in range(n):
        pivot = None
        for r in range(rank, n):
            if A[r][col]:
                pivot = r
                break
        if pivot is None:
            continue
        A[rank], A[pivot] = A[pivot], A[rank]
        for r in range(n):
            if r != rank and A[r][col]:
                for c in range(col, n):
                    A[r][c] ^= A[rank][c]
        rank += 1
    return rank == n

def random_invertible_matrix():
    while True:
        M = [[random.randint(0, 1) for _ in range(8)] for _ in range(8)]
        if matrix_is_invertible(M):
            return M

# ========== S-box Construction ==========
def construct_sbox(K, c, mult_inv_matrix):
    sbox = [0] * 256
    c_bits = byte_to_bits(c)
    for x in range(256):
        row = x // 16
        col = x % 16
        inv = mult_inv_matrix[row][col]
        inv_bits = byte_to_bits(inv)
        transformed = matrix_vec_mul(K, inv_bits)
        for i in range(8):
            transformed[i] ^= c_bits[i]
        sbox[x] = bits_to_byte(transformed)
    return sbox

def invert_sbox(sbox):
    inv = [0] * 256
    for i, v in enumerate(sbox):
        inv[v] = i
    return inv

# ========== S-box Testing ==========
def test_balance(sbox):
    for bit_pos in range(8):
        count_0 = 0
        count_1 = 0
        for x in range(256):
            bit_val = (sbox[x] >> bit_pos) & 1
            if bit_val == 0:
                count_0 += 1
            else:
                count_1 += 1
        if count_0 != 128 or count_1 != 128:
            return False
    return True

def test_bijective(sbox):
    return len(set(sbox)) == 256

# ========== Fast Walsh-Hadamard Transform ==========
def fwht(a):
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h*2):
            for j in range(i, i+h):
                x = a[j]
                y = a[j+h]
                a[j] = x + y
                a[j+h] = x - y
        h *= 2

def walsh_spectrum_from_truth(truth):
    vec = [1 if v == 0 else -1 for v in truth]
    fw = vec[:]
    fwht(fw)
    return fw

# ========== Cryptographic Metrics ==========
def nonlinearity_of_sbox(sbox):
    best = 256
    for a in range(1, 256):
        truth = [(bin(a & sbox[x]).count("1") & 1) for x in range(256)]
        W = walsh_spectrum_from_truth(truth)
        maxW = max(abs(v) for v in W)
        nl = 128 - (maxW // 2)
        if nl < best:
            best = nl
    return best

def sac_of_sbox(sbox):
    n = 8
    total = 0
    for i in range(n):
        for x in range(256):
            y = x ^ (1 << i)
            diff = sbox[x] ^ sbox[y]
            total += bin(diff).count("1")
    avg = total / (256 * 8 * 8)
    return avg

def bic_nl_of_sbox(sbox):
    best = 256
    for b in range(8):
        truth = [(sbox[x] >> b) & 1 for x in range(256)]
        W = walsh_spectrum_from_truth(truth)
        maxW = max(abs(v) for v in W)
        nl = 128 - (maxW // 2)
        if nl < best:
            best = nl
    for b1 in range(8):
        for b2 in range(b1 + 1, 8):
            mask = (1 << b1) | (1 << b2)
            truth = [(bin(mask & sbox[x]).count("1") & 1) for x in range(256)]
            W = walsh_spectrum_from_truth(truth)
            maxW = max(abs(v) for v in W)
            nl = 128 - (maxW // 2)
            if nl < best:
                best = nl
    return best

def bic_sac_of_sbox(sbox):
    best_diff = 1.0
    best_avg = None
    for mask in range(1, 256):
        total = 0
        for i in range(8):
            for x in range(256):
                y = x ^ (1 << i)
                fx = bin(mask & sbox[x]).count("1") & 1
                fy = bin(mask & sbox[y]).count("1") & 1
                if fx != fy:
                    total += 1
        avg = total / (256 * 8)
        diff = abs(avg - 0.5)
        if diff < best_diff:
            best_diff = diff
            best_avg = avg
    return best_avg

def lap_of_sbox(sbox):
    max_abs_W = 0
    for b in range(1, 256):
        truth = [(bin(b & sbox[x]).count("1") & 1) for x in range(256)]
        W = walsh_spectrum_from_truth(truth)
        local_max = max(abs(v) for v in W)
        if local_max > max_abs_W:
            max_abs_W = local_max
    bias = max_abs_W / 512.0
    return bias

def dap_of_sbox(sbox):
    max_count = 0
    for dx in range(1, 256):
        counter = Counter()
        for x in range(256):
            dy = sbox[x] ^ sbox[x ^ dx]
            counter[dy] += 1
        local_max = max(counter.values())
        if local_max > max_count:
            max_count = local_max
    dap = max_count / 256.0
    return dap

def evaluate_sbox(sbox):
    return {
        'NL': nonlinearity_of_sbox(sbox),
        'SAC': sac_of_sbox(sbox),
        'BIC-NL': bic_nl_of_sbox(sbox),
        'BIC-SAC': bic_sac_of_sbox(sbox),
        'LAP': lap_of_sbox(sbox),
        'DAP': dap_of_sbox(sbox)
    }

# ========== Standard AES S-box and Constant ==========
K_AES = [
    [1,0,0,0,1,1,1,1],
    [1,1,0,0,0,1,1,1],
    [1,1,1,0,0,0,1,1],
    [1,1,1,1,0,0,0,1],
    [1,1,1,1,1,0,0,0],
    [0,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,0],
    [0,0,0,1,1,1,1,1]
]
C_AES = 0b11000110  # constant used for affine

standard_sbox = [
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
]

# ========== MAIN EXECUTION ==========
def main():
    print("="*80)
    print(" "*20 + "AES S-BOX MODIFICATION")
    print(" "*15 + "Paper Implementation & Analysis")
    print("="*80)
    
    mult_inv_matrix = generate_multiplicative_inverse_matrix()
    
    std_metrics = evaluate_sbox(standard_sbox)
    
    SAMPLES = 128
    valid_candidates = []

    for t in range(SAMPLES):
        K = random_invertible_matrix()
        c = C_AES
        sbox = construct_sbox(K, c, mult_inv_matrix)
        if not test_balance(sbox):
            continue
        if not test_bijective(sbox):
            continue
        metrics = evaluate_sbox(sbox)
        s_value = (abs(metrics['SAC'] - 0.5) + abs(metrics['BIC-SAC'] - 0.5)) / 2
        sv_value = (120 - metrics['NL']) + abs(0.5 - metrics['SAC']) + \
                   (120 - metrics['BIC-NL']) + abs(0.5 - metrics['BIC-SAC'])
        valid_candidates.append({
            's_value': s_value,
            'sv_value': sv_value,
            'K': K,
            'c': c,
            'sbox': sbox,
            'metrics': metrics,
            'orig_index': t
        })

    if not valid_candidates:
        print("No valid candidates found. Exiting.")
        return

    valid_candidates.sort(key=lambda x: (x['s_value'], x['sv_value']))
    best = valid_candidates[0]

    print(f"\nProposed S-box terbaik adalah kandidat ke-{best['orig_index']} (orig_index).")
    
    # ========== SIMPAN HASIL S-BOX ==========
    # Save decimal values
    with open("best_sbox_decimal.txt", "w") as f:
        for i in range(16):
            row = best['sbox'][i*16:(i+1)*16]
            f.write(" ".join(str(x) for x in row) + "\n")
    
    # Save hex values
    with open("best_sbox_hex.txt", "w") as f:
        for i in range(16):
            row = best['sbox'][i*16:(i+1)*16]
            f.write(" ".join(f"0x{x:02X}" for x in row) + "\n")
    
    print("✓ Best S-box saved to 'best_sbox_decimal.txt' and 'best_sbox_hex.txt'")

if __name__ == "__main__":
    main()
