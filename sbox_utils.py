import numpy as np
import csv
import json

# ================= CONSTANT =================
C = np.array([1,1,0,0,0,1,1,0], dtype=np.uint8)

# ================= MULTIPLICATIVE INVERSE TABLE =================
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

# ================= AFFINE MATRICES =================
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

# K4 from paper (Table 4)
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

# K128 from paper (Table 8)
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

# ================= CONSTRUCT SBOX =================
def construct_sbox(A):
    s = np.zeros(256, dtype=np.uint8)
    for x in range(256):
        inv = INV[x//16, x%16]
        bits = np.array([(inv >> i) & 1 for i in range(8)], dtype=np.uint8)
        out = (A @ bits + C) % 2
        s[x] = sum(int(out[i]) << i for i in range(8))
    return s

def generate_sboxes(include_random=True, seed=None):
    sboxes = {
        "A0": construct_sbox(A0),
        "A1": construct_sbox(A1),
        "A2": construct_sbox(A2),
        "K4": construct_sbox(K4),
        "K44": construct_sbox(K44),
        "K128": construct_sbox(K128)
    }
    AES_SBOX = np.array([
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
    sboxes["AES"] = AES_SBOX

    if include_random:
        if seed is not None:
            np.random.seed(seed)
        sboxes["RANDOM"] = np.random.permutation(256).astype(np.uint8)

    return sboxes

# ================= METRICS =================
def balance(s):
    return all(np.sum((s >> b) & 1) == 128 for b in range(8))

def bijective(s):
    return len(set(s)) == 256

def nonlinearity(s):
    nl = 256
    for a in range(1,256):
        for b in range(1,256):
            cnt = sum(((bin(x & a).count("1") ^ bin(s[x] & b).count("1")) % 2 == 0) for x in range(256))
            nl = min(nl, 128 - abs(cnt - 128))
    return nl

def sac(s):
    total = 0
    for x in range(256):
        for i in range(8):
            total += bin(s[x] ^ s[x ^ (1<<i)]).count("1")
    return total / (256*8*8)

def lap(s):
    max_bias = 0
    for a in range(1,256):
        for b in range(1,256):
            cnt = sum(((bin(x & a).count("1") ^ bin(s[x] & b).count("1")) % 2 == 0) for x in range(256))
            max_bias = max(max_bias, abs(cnt-128)/256)
    return max_bias**2

def dap(s):
    table = np.zeros((256,256), dtype=int)
    for x in range(256):
        for dx in range(1,256):
            dy = s[x] ^ s[x^dx]
            table[dx,dy] += 1
    return np.max(table[1:])/256

def bic_sac_fast(s):
    bic_total = 0
    count = 0
    s_arr = np.array([[(s[x]^s[x^(1<<i)])>>np.arange(8)&1 for i in range(8)] for x in range(256)])
    for i in range(8):
        for j in range(i+1,8):
            diff1 = s_arr[:,i,:]
            diff2 = s_arr[:,j,:]
            bic_total += np.sum(diff1[:,:,None]^diff2[:,None,:])
            count += diff1.size*diff2.shape[1]
    return bic_total/count

def bic_nl_fast(s):
    s_bits = np.array([(s[x]>>np.arange(8))&1 for x in range(256)])
    max_bias = 0
    for a1 in range(8):
        for a2 in range(a1+1,8):
            for b1 in range(8):
                for b2 in range(b1+1,8):
                    f1 = np.array([(x>>a1&1)^s_bits[x,b1] for x in range(256)])
                    f2 = np.array([(x>>a2&1)^s_bits[x,b2] for x in range(256)])
                    cnt = np.sum(f1==f2)
                    max_bias = max(max_bias, abs(cnt-128))
    return 128-max_bias

# ================= EVALUATE & SAVE =================
def evaluate_sbox(name, s):
    return {
        "S-box": name,
        "Balance": balance(s),
        "Bijective": bijective(s),
        "NL": nonlinearity(s),
        "SAC": sac(s),
        "LAP": lap(s),
        "DAP": dap(s),
        "BIC-SAC": bic_sac_fast(s),
        "BIC-NL": bic_nl_fast(s)
    }

def save_results_csv(results, filename="sbox_results.csv"):
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(results.values())).keys()))
        writer.writeheader()
        for r in results.values():
            writer.writerow(r)

def save_results_json(results, filename="sbox_results.json"):
    # Convert numpy types to Python native types
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    results_native = convert_to_native(results)
    with open(filename, "w") as f:
        json.dump(results_native, f, indent=4)

# ================= DETAILED TEST PROCESS =================
def show_test_process(name, s, verbose=True):
    """Show detailed testing process for an S-box"""
    if verbose:
        print(f"\n{'='*80}")
        print(f"TESTING S-BOX: {name}")
        print(f"{'='*80}\n")
    
    results = {}
    
    # Test 1: Balance
    if verbose:
        print("📊 [1/8] Testing BALANCE...")
        print("   → Checking if each output bit has equal 0s and 1s (128 each)")
    balance_result = balance(s)
    results['Balance'] = balance_result
    if verbose:
        status = "✅ PASS" if balance_result else "❌ FAIL"
        print(f"   Result: {status} - {balance_result}\n")
    
    # Test 2: Bijective
    if verbose:
        print("🔄 [2/8] Testing BIJECTIVE...")
        print("   → Checking if all 256 values are unique (one-to-one mapping)")
        print(f"   Unique values: {len(set(s))}/256")
    bijective_result = bijective(s)
    results['Bijective'] = bijective_result
    if verbose:
        status = "✅ PASS" if bijective_result else "❌ FAIL"
        print(f"   Result: {status} - {bijective_result}\n")
    
    # Test 3: Nonlinearity (NL)
    if verbose:
        print("📈 [3/8] Testing NONLINEARITY (NL)...")
        print("   → Computing minimum Hamming distance from affine functions")
        print("   → Iterating through 65,025 combinations (255 x 255)...")
    nl_result = nonlinearity(s)
    results['NL'] = nl_result
    if verbose:
        print(f"   Result: {nl_result} (Ideal: 112, Max: 120)")
        quality = "Excellent" if nl_result >= 112 else "Good" if nl_result >= 104 else "Moderate"
        print(f"   Quality: {quality}\n")
    
    # Test 4: Strict Avalanche Criterion (SAC)
    if verbose:
        print("🌊 [4/8] Testing STRICT AVALANCHE CRITERION (SAC)...")
        print("   → Testing avalanche effect: 1-bit input change affects output")
        print("   → Computing for 256 inputs x 8 bit positions = 2,048 tests")
    sac_result = sac(s)
    results['SAC'] = sac_result
    if verbose:
        deviation = abs(0.5 - sac_result)
        print(f"   Result: {sac_result:.6f} (Ideal: 0.5)")
        print(f"   Deviation from ideal: {deviation:.6f}")
        quality = "Excellent" if deviation < 0.01 else "Good" if deviation < 0.02 else "Moderate"
        print(f"   Quality: {quality}\n")
    
    # Test 5: Linear Approximation Probability (LAP)
    if verbose:
        print("🔍 [5/8] Testing LINEAR APPROXIMATION PROBABILITY (LAP)...")
        print("   → Measuring resistance to linear cryptanalysis")
        print("   → Testing 65,025 linear approximations...")
    lap_result = lap(s)
    results['LAP'] = lap_result
    if verbose:
        print(f"   Result: {lap_result:.6f} (Lower is better, Ideal: 0.0625)")
        quality = "Excellent" if lap_result <= 0.0625 else "Good" if lap_result <= 0.075 else "Moderate"
        print(f"   Quality: {quality}\n")
    
    # Test 6: Differential Approximation Probability (DAP)
    if verbose:
        print("🎯 [6/8] Testing DIFFERENTIAL APPROXIMATION PROBABILITY (DAP)...")
        print("   → Measuring resistance to differential cryptanalysis")
        print("   → Building differential distribution table (256x256)...")
    dap_result = dap(s)
    results['DAP'] = dap_result
    if verbose:
        print(f"   Result: {dap_result:.6f} (Lower is better, Ideal: ≤ 0.015625)")
        quality = "Excellent" if dap_result <= 0.015625 else "Good" if dap_result <= 0.02 else "Moderate"
        print(f"   Quality: {quality}\n")
    
    # Test 7: BIC-SAC
    if verbose:
        print("🔗 [7/8] Testing BIT INDEPENDENCE - SAC (BIC-SAC)...")
        print("   → Testing independence between output bit changes")
        print("   → Analyzing 28 bit pairs (8 choose 2)...")
    bic_sac_result = bic_sac_fast(s)
    results['BIC-SAC'] = bic_sac_result
    if verbose:
        deviation = abs(0.5 - bic_sac_result)
        print(f"   Result: {bic_sac_result:.6f} (Ideal: 0.5)")
        print(f"   Deviation from ideal: {deviation:.6f}")
        quality = "Excellent" if deviation < 0.01 else "Good" if deviation < 0.02 else "Moderate"
        print(f"   Quality: {quality}\n")
    
    # Test 8: BIC-NL
    if verbose:
        print("🧩 [8/8] Testing BIT INDEPENDENCE - NONLINEARITY (BIC-NL)...")
        print("   → Testing nonlinearity between output bit pairs")
        print("   → Computing for all bit pair combinations...")
    bic_nl_result = bic_nl_fast(s)
    results['BIC-NL'] = bic_nl_result
    if verbose:
        print(f"   Result: {bic_nl_result} (Ideal: 112, Max: 120)")
        quality = "Excellent" if bic_nl_result >= 112 else "Good" if bic_nl_result >= 104 else "Moderate"
        print(f"   Quality: {quality}\n")
    
    # Summary
    if verbose:
        print(f"{'='*80}")
        print(f"SUMMARY FOR S-BOX: {name}")
        print(f"{'='*80}")
        print(f"Balance:   {results['Balance']}")
        print(f"Bijective: {results['Bijective']}")
        print(f"NL:        {results['NL']}")
        print(f"SAC:       {results['SAC']:.6f}")
        print(f"LAP:       {results['LAP']:.6f}")
        print(f"DAP:       {results['DAP']:.6f}")
        print(f"BIC-SAC:   {results['BIC-SAC']:.6f}")
        print(f"BIC-NL:    {results['BIC-NL']}")
        print(f"{'='*80}\n")
    
    return results

def compare_with_ideal(results):
    """Compare results with ideal values and show quality assessment"""
    print("\n" + "="*80)
    print("QUALITY ASSESSMENT (Comparison with Ideal Values)")
    print("="*80)
    
    # Define ideal values and scoring
    ideals = {
        'Balance': (True, 'boolean'),
        'Bijective': (True, 'boolean'),
        'NL': (112, 'higher'),
        'SAC': (0.5, 'closer'),
        'LAP': (0.0625, 'lower_equal'),
        'DAP': (0.015625, 'lower_equal'),
        'BIC-SAC': (0.5, 'closer'),
        'BIC-NL': (112, 'higher')
    }
    
    scores = []
    
    for metric, (ideal, comparison_type) in ideals.items():
        value = results[metric]
        
        if comparison_type == 'boolean':
            score = 100 if value == ideal else 0
            status = "✅ PASS" if value == ideal else "❌ FAIL"
            detail = f"{value}"
        
        elif comparison_type == 'higher':
            score = min(100, (value / ideal) * 100) if ideal > 0 else 0
            status = "✅ Excellent" if value >= ideal else "⚠️ Good" if value >= ideal * 0.9 else "❌ Needs Improvement"
            detail = f"{value} / {ideal}"
        
        elif comparison_type == 'closer':
            deviation = abs(value - ideal)
            score = max(0, 100 - (deviation * 1000))  # Penalize deviation
            status = "✅ Excellent" if deviation < 0.01 else "⚠️ Good" if deviation < 0.02 else "❌ Needs Improvement"
            detail = f"{value:.6f} (Ideal: {ideal}, Deviation: {deviation:.6f})"
        
        elif comparison_type == 'lower_equal':
            if value <= ideal:
                score = 100
                status = "✅ Excellent"
            else:
                score = max(0, 100 - ((value - ideal) / ideal * 100))
                status = "⚠️ Acceptable" if value <= ideal * 1.2 else "❌ Needs Improvement"
            detail = f"{value:.6f} (Ideal: ≤ {ideal})"
        
        scores.append(score)
        print(f"{metric:12} {status:20} {detail}")
    
    overall_score = sum(scores) / len(scores)
    print(f"\n{'='*80}")
    print(f"OVERALL SCORE: {overall_score:.2f}/100")
    
    if overall_score >= 90:
        grade = "A+ (Outstanding)"
    elif overall_score >= 80:
        grade = "A (Excellent)"
    elif overall_score >= 70:
        grade = "B (Good)"
    elif overall_score >= 60:
        grade = "C (Acceptable)"
    else:
        grade = "D (Needs Improvement)"
    
    print(f"GRADE: {grade}")
    print(f"{'='*80}\n")
    
    return overall_score

# ================= MAIN =================
if __name__=="__main__":
    print("\n" + "="*80)
    print("S-BOX CRYPTOGRAPHIC STRENGTH TESTING SUITE")
    print("="*80)
    print("This program tests S-boxes against standard cryptographic criteria:")
    print("  • Balance: Equal distribution of 0s and 1s")
    print("  • Bijectivity: One-to-one mapping (all unique values)")
    print("  • Nonlinearity (NL): Resistance to linear approximation")
    print("  • SAC: Avalanche effect (bit changes propagate)")
    print("  • LAP: Linear approximation probability")
    print("  • DAP: Differential approximation probability")
    print("  • BIC-SAC: Bit independence in avalanche")
    print("  • BIC-NL: Bit independence in nonlinearity")
    print("="*80 + "\n")
    
    sboxes = generate_sboxes(include_random=True, seed=42)
    all_results = {}
    
    # Test each S-box with detailed process
    for name, s in sboxes.items():
        results = show_test_process(name, s, verbose=True)
        all_results[name] = results
        
        # Show quality assessment
        compare_with_ideal(results)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS...")
    print("="*80)
    
    # Prepare results for CSV/JSON
    formatted_results = {}
    for name, results in all_results.items():
        formatted_results[name] = {
            "S-box": name,
            **results
        }
    
    save_results_csv(formatted_results, "sbox_results.csv")
    save_results_json(formatted_results, "sbox_results.json")
    
    print("✅ Results saved to:")
    print("   • sbox_results.csv")
    print("   • sbox_results.json")
    print("="*80 + "\n")
    
    # Final comparison table
    print("\n" + "="*80)
    print("FINAL COMPARISON TABLE")
    print("="*80)
    print(f"{'S-box':<10} {'Balance':<8} {'Bijec':<8} {'NL':<6} {'SAC':<10} {'LAP':<10} {'DAP':<10} {'BIC-SAC':<10} {'BIC-NL':<6}")
    print("-"*80)
    
    for name, results in all_results.items():
        print(f"{name:<10} {str(results['Balance']):<8} {str(results['Bijective']):<8} "
              f"{results['NL']:<6} {results['SAC']:<10.6f} {results['LAP']:<10.6f} "
              f"{results['DAP']:<10.6f} {results['BIC-SAC']:<10.6f} {results['BIC-NL']:<6}")
    
    print("="*80 + "\n")