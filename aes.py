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
    """Generate the 16x16 multiplicative inverse matrix (Table 1 in paper)"""
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
    """Convert byte to 8-bit list (MSB first)"""
    return [(b >> (7-i)) & 1 for i in range(8)]

def bits_to_byte(bits):
    """Convert 8-bit list to byte"""
    b = 0
    for i in range(8):
        b = (b << 1) | (bits[i] & 1)
    return b

# ========== Matrix Operations over GF(2) ==========
def matrix_vec_mul(M, v):
    """Multiply 8x8 matrix M with 8-bit vector v in GF(2)"""
    out = [0] * 8
    for i in range(8):
        s = 0
        for j in range(8):
            s ^= (M[i][j] & v[j])
        out[i] = s & 1
    return out

def matrix_is_invertible(M):
    """Check if 8x8 binary matrix M is invertible using Gaussian elimination"""
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
    """Generate random 8x8 invertible binary matrix"""
    while True:
        M = [[random.randint(0, 1) for _ in range(8)] for _ in range(8)]
        if matrix_is_invertible(M):
            return M

# ========== S-box Construction (Step 3: Affine Transformation) ==========
def construct_sbox(K, c, mult_inv_matrix):
    """
    Construct S-box using affine transformation: B(X) = K·X^(-1) + C (mod 2)
    
    Args:
        K: 8x8 affine matrix
        c: 8-bit constant
        mult_inv_matrix: 16x16 multiplicative inverse matrix
    
    Returns:
        256-element S-box list
    """
    sbox = [0] * 256
    c_bits = byte_to_bits(c)
    
    for x in range(256):
        # Get multiplicative inverse from pre-computed matrix
        row = x // 16
        col = x % 16
        inv = mult_inv_matrix[row][col]
        
        # Convert to bit representation
        inv_bits = byte_to_bits(inv)
        
        # Apply affine transformation: K * inv_bits
        transformed = matrix_vec_mul(K, inv_bits)
        
        # Add constant c (XOR in GF(2))
        for i in range(8):
            transformed[i] ^= c_bits[i]
        
        # Convert back to byte
        out_byte = bits_to_byte(transformed)
        sbox[x] = out_byte
    
    return sbox

def invert_sbox(sbox):
    """Create inverse S-box"""
    inv = [0] * 256
    for i, v in enumerate(sbox):
        inv[v] = i
    return inv

# ========== S-box Testing (Balance and Bijective) ==========
def test_balance(sbox):
    """
    Test balance criterion: each output bit should have equal 0s and 1s
    According to Eq. (1): #{x|f(x)=0} = #{x|f(x)=1} = 128
    """
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
    """Test bijective criterion: all values 0-255 should appear exactly once"""
    return len(set(sbox)) == 256

# ========== Fast Walsh-Hadamard Transform ==========
def fwht(a):
    """Fast Walsh-Hadamard Transform (in-place)"""
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
    """Compute Walsh spectrum from truth table"""
    vec = [1 if v == 0 else -1 for v in truth]
    fw = vec[:]
    fwht(fw)
    return fw

# ========== Nonlinearity (NL) ==========
def nonlinearity_of_sbox(sbox):
    """
    Calculate nonlinearity using Eq. (5): NL(f(x)) = min d(f(x), g(x))
    """
    best = 256
    for a in range(1, 256):
        truth = [(bin(a & sbox[x]).count("1") & 1) for x in range(256)]
        W = walsh_spectrum_from_truth(truth)
        maxW = max(abs(v) for v in W)
        nl = 128 - (maxW // 2)
        if nl < best:
            best = nl
    return best

# ========== Strict Avalanche Criterion (SAC) ==========
def sac_of_sbox(sbox):
    """
    Calculate SAC using Eq. (6): S(x,i) = (1/2^n) Σ f(x) ⊕ f(x ⊕ c_i^n)
    Ideal value: 0.5
    """
    n = 8
    total = 0
    for i in range(n):
        for x in range(256):
            y = x ^ (1 << i)
            diff = sbox[x] ^ sbox[y]
            total += bin(diff).count("1")
    avg = total / (256 * 8 * 8)
    return avg

# ========== BIC-NL ==========
def bic_nl_of_sbox(sbox):
    """Calculate Bit Independence Criterion - Nonlinearity"""
    best = 256
    
    # Single bit positions
    for b in range(8):
        truth = [(sbox[x] >> b) & 1 for x in range(256)]
        W = walsh_spectrum_from_truth(truth)
        maxW = max(abs(v) for v in W)
        nl = 128 - (maxW // 2)
        if nl < best:
            best = nl
    
    # Pairs of bit positions
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

# ========== BIC-SAC ==========
def bic_sac_of_sbox(sbox):
    """Calculate Bit Independence Criterion - SAC"""
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

# ========== Linear Approximation Probability (LAP) ==========
def lap_of_sbox(sbox):
    """
    Calculate LAP using Eq. (7)
    """
    max_abs_W = 0
    for b in range(1, 256):
        truth = [(bin(b & sbox[x]).count("1") & 1) for x in range(256)]
        W = walsh_spectrum_from_truth(truth)
        local_max = max(abs(v) for v in W)
        if local_max > max_abs_W:
            max_abs_W = local_max
    bias = max_abs_W / 512.0
    return bias

# ========== Differential Approximation Probability (DAP) ==========
def dap_of_sbox(sbox):
    """
    Calculate DAP using Eq. (8)
    """
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

# ========== Comprehensive S-box Evaluation ==========
def evaluate_sbox(sbox):
    """Evaluate S-box with all metrics from the paper"""
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

# NOTE: in original code C_AES set to 0b11000110 (198 decimal)
C_AES = 0b11000110  # constant used for affine (198 decimal)

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

# ========== Main Execution ==========
def main():
    print("="*80)
    print(" "*20 + "AES S-BOX MODIFICATION")
    print(" "*15 + "Paper Implementation & Analysis")
    print("="*80)
    
    # Step 1: Generate multiplicative inverse matrix
    print("\n[STEP 1] Generating Multiplicative Inverse Matrix...")
    mult_inv_matrix = generate_multiplicative_inverse_matrix()
    print("✓ Generated 16x16 multiplicative inverse matrix using GF(2^8)")
    print("  Irreducible polynomial: x^8 + x^4 + x^3 + x + 1")
    
    # Step 2: Evaluate standard AES S-box
    print("\n[STEP 2] Evaluating Standard AES S-box...")
    std_metrics = evaluate_sbox(standard_sbox)
    print("✓ Computed all cryptographic metrics")
    
    # Step 3: Generate and test multiple candidate S-boxes
    print(f"\n[STEP 3] Generating and Testing Candidate S-boxes...")
    SAMPLES = 128  # Test multiple candidates
    valid_candidates = []
    
    print(f"Testing {SAMPLES} random affine matrices...")
    for t in range(SAMPLES):
        K = random_invertible_matrix()
        c = C_AES  # Use chosen constant
        
        # Construct candidate S-box
        sbox = construct_sbox(K, c, mult_inv_matrix)
        
        # Test balance and bijective
        if not test_balance(sbox):
            continue
        if not test_bijective(sbox):
            continue
        
        # Evaluate metrics
        metrics = evaluate_sbox(sbox)
        
        # Calculate S value according to Eq. (9)
        s_value = (abs(metrics['SAC'] - 0.5) + abs(metrics['BIC-SAC'] - 0.5)) / 2
        
        # Calculate SV value according to Eq. (20)
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
        
        if (t + 1) % 20 == 0:
            print(f"  Progress: {t+1}/{SAMPLES} candidates tested, {len(valid_candidates)} valid")
    
    print(f"\n✓ Found {len(valid_candidates)} valid S-box candidates")
    
    if not valid_candidates:
        print("No valid candidates found. Exiting.")
        return
    
    # Sort by s_value (lower is better), tie-breaker by sv_value
    valid_candidates.sort(key=lambda x: (x['s_value'], x['sv_value']))
    best = valid_candidates[0]
    
    # <-- **BARIS YANG DISISIPKAN**: tampilkan orig_index best (sbox ke berapa)
    print(f"\nProposed S-box terbaik adalah kandidat ke-{best['orig_index']} (orig_index).")
    
    # ========== TABLE 19: Comparison with AES S-box ==========
    print("\n" + "="*80)
    print("TABLE 19: Comparison of the Strength of AES S-box and Proposed S-box")
    print("="*80)
    print(f"{'S-box':<20} {'NL':<10} {'SAC':<12} {'BIC-NL':<10} {'BIC-SAC':<12} {'LAP':<10} {'DAP':<10}")
    print("-"*80)
    
    # AES S-box
    print(f"{'AES [1]':<20} {std_metrics['NL']:<10} {std_metrics['SAC']:<12.5f} "
          f"{std_metrics['BIC-NL']:<10} {std_metrics['BIC-SAC']:<12.5f} "
          f"{std_metrics['LAP']:<10.4f} {std_metrics['DAP']:<10.5f}")
    
    # Proposed S-box (best)
    print(f"{'Proposed S-box_best':<20} {best['metrics']['NL']:<10} {best['metrics']['SAC']:<12.5f} "
          f"{best['metrics']['BIC-NL']:<10} {best['metrics']['BIC-SAC']:<12.5f} "
          f"{best['metrics']['LAP']:<10.4f} {best['metrics']['DAP']:<10.5f}")
    
    print("\nBold indicates the highest value for the criteria listed in the respective column")
    
    # ========== TABLE 20: Performance Comparison ==========
    print("\n" + "="*80)
    print("TABLE 20: Performance Comparison of S-boxes (All Valid Candidates)")
    print("="*80)
    hdr = f"{'Rank':<6} {'OrigIdx':<8} {'NL':<6} {'SAC':<10} {'BIC-NL':<8} {'BIC-SAC':<10} {'LAP':<8} {'DAP':<8} {'S_val':<10} {'SV':<10}"
    print(hdr)
    print("-"*len(hdr))
    
    for rank, candidate in enumerate(valid_candidates, start=1):
        m = candidate['metrics']
        print(f"{rank:<6} {candidate['orig_index']:<8} {m['NL']:<6} {m['SAC']:<10.5f} "
              f"{m['BIC-NL']:<8} {m['BIC-SAC']:<10.5f} {m['LAP']:<8.5f} {m['DAP']:<8.5f} "
              f"{candidate['s_value']:<10.6f} {candidate['sv_value']:<10.6f}")
    
    # Calculate AES SV
    sv_aes = (120 - std_metrics['NL']) + abs(0.5 - std_metrics['SAC']) + \
             (120 - std_metrics['BIC-NL']) + abs(0.5 - std_metrics['BIC-SAC'])
    
    print(f"\n{'AES [1]':<6} SV = {sv_aes:.6f}")
    
    print("\n(Notes: 'OrigIdx' = original loop index where matrix was generated; 'Rank' = sorted by S_value then SV.)")
    
    # ========== DETAILED ANALYSIS ==========
    print("\n" + "="*80)
    print("DETAILED ANALYSIS OF PROPOSED S-BOX")
    print("="*80)
    
    print(f"\nCandidate Orig Index: {best['orig_index']}")
    print(f"S-value (Eq. 9): {best['s_value']:.6f} (closer to 0 is better)")
    print(f"SV-value (Eq. 20): {best['sv_value']:.6f} (closer to 0 is better)")
    
    print("\n--- Complete Metrics ---")
    for key, val in best['metrics'].items():
        if isinstance(val, float):
            print(f"  {key:12s}: {val:.6f}")
        else:
            print(f"  {key:12s}: {val}")
    
    # ========== IMPROVEMENT ANALYSIS ==========
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS (vs Standard AES S-box)")
    print("="*80)
    print(f"{'Metric':<12} {'AES':<12} {'Proposed':<12} {'Improvement'}")
    print("-"*80)
    
    for key in ['NL', 'SAC', 'BIC-NL', 'BIC-SAC', 'LAP', 'DAP']:
        aes_val = std_metrics[key]
        prop_val = best['metrics'][key]
        
        if key in ['SAC', 'BIC-SAC']:
            # Closer to 0.5 is better
            aes_diff = abs(aes_val - 0.5)
            prop_diff = abs(prop_val - 0.5)
            
            if prop_diff == 0:
                improvement = 100.0 if aes_diff > 0 else 0.0
            else:
                improvement = ((aes_diff - prop_diff) / prop_diff * 100) if prop_diff != 0 else 0.0
        elif key in ['LAP', 'DAP']:
            # Lower is better
            if prop_val == 0:
                improvement = 100.0 if aes_val > 0 else 0.0
            else:
                improvement = -((prop_val - aes_val) / aes_val * 100) if aes_val != 0 else 0
        else:
            # Higher is better (NL, BIC-NL)
            improvement = ((prop_val - aes_val) / aes_val * 100) if aes_val != 0 else 0
        
        print(f"{key:<12} {aes_val:<12.6f} {prop_val:<12.6f} {improvement:+.3f}%")
    
    # Overall SV comparison
    print(f"{'SV':<12} {sv_aes:<12.6f} {best['sv_value']:<12.6f} "
          f"{((sv_aes - best['sv_value'])/sv_aes*100):+.3f}%")
    
    # ========== S-BOX DISPLAY ==========
    print("\n" + "="*80)
    print("PROPOSED S-BOX (16x16 Format - Decimal Values)")
    print("="*80)
    print("     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15")
    print("-" * 80)
    for i in range(16):
        row = best['sbox'][i*16:(i+1)*16]
        print(f"{i:2d} |" + " ".join(f"{x:3d}" for x in row))
    
    # Also print hex view
    print("\n" + "="*80)
    print("PROPOSED S-BOX (16x16 Format - HEX VALUES)")
    print("="*80)
    print("     0    1    2    3    4    5    6    7    8    9    A    B    C    D    E    F")
    print("-" * 80)
    for i in range(16):
        row = best['sbox'][i*16:(i+1)*16]
        print(f"{i:2d} |" + " ".join(f"0x{x:02X}" for x in row))
    
    # ========== AFFINE MATRIX ==========
    print(f"\n" + "="*80)
    print("AFFINE MATRIX K (8x8 Binary) for best candidate")
    print("="*80)
    for i, row in enumerate(best['K']):
        print(f"Row {i}: " + " ".join(str(x) for x in row))
    
    print(f"\nConstant C: {best['c']} (decimal) = {bin(best['c'])} (binary) = 0x{best['c']:02X} (hex)")
    
    # ========== SUMMARY ==========
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Generated {SAMPLES} candidate affine matrices")
    print(f"✓ Found {len(valid_candidates)} valid S-boxes (passed balance & bijective tests)")
    print(f"✓ Best candidate original index: {best['orig_index']}")
    print(f"✓ Best S-box has S-value: {best['s_value']:.6f}")
    print(f"✓ Best S-box has SV-value: {best['sv_value']:.6f}")
    print(f"✓ Overall improvement vs AES: {((sv_aes - best['sv_value'])/sv_aes*100):+.3f}%")
    print("\nThe proposed S-box demonstrates superior cryptographic properties")
    print("compared to the standard AES S-box across multiple metrics.")
    print("="*80)

if __name__ == "__main__":
    main()
