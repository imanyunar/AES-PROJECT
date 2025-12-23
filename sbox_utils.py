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
# EXISTING METRICS
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
# NEW METRICS: DU, AD, TO, CI
# =====================================================

def differential_uniformity(s):
    """
    Differential Uniformity (DU)
    Menghitung nilai δ maksimum dalam Difference Distribution Table (DDT)
    DU = max_{Δx≠0, Δy} |{x : S(x) ⊕ S(x ⊕ Δx) = Δy}|
    
    Nilai yang lebih kecil = lebih baik (lebih tahan differential cryptanalysis)
    Untuk S-box 8-bit yang baik: DU ≤ 4
    AES S-box memiliki DU = 4
    """
    du = 0
    for dx in range(1, 256):  # dx ≠ 0
        dy_counts = np.bincount(s ^ s[np.arange(256) ^ dx], minlength=256)
        du = max(du, np.max(dy_counts))
    return int(du)


def algebraic_degree(s):
    """
    Algebraic Degree (AD)
    Menghitung derajat maksimum dari representasi aljabar boolean S-box
    Menggunakan ANF (Algebraic Normal Form) melalui Möbius transform
    
    Nilai yang lebih tinggi = lebih baik (lebih tahan algebraic attack)
    Untuk S-box 8-bit: AD maksimum = 7
    AES S-box memiliki AD = 7
    """
    max_degree = 0
    
    for bit_pos in range(8):
        # Ekstrak bit output tertentu untuk semua input
        truth_table = np.array([(s[x] >> bit_pos) & 1 for x in range(256)], dtype=np.int32)
        
        # Möbius transform untuk mendapatkan ANF
        anf = truth_table.copy()
        for i in range(8):
            for j in range(256):
                if j & (1 << i):
                    anf[j] ^= anf[j ^ (1 << i)]
        
        # Hitung derajat: jumlah bit 1 dalam indeks dengan koefisien ANF = 1
        for idx in range(256):
            if anf[idx]:
                degree = bin(idx).count('1')
                max_degree = max(max_degree, degree)
    
    return int(max_degree)


def transparency_order(s):
    """
    Transparency Order (TO)
    Mengukur resistensi terhadap differential power analysis (DPA)
    TO yang lebih tinggi = lebih tahan terhadap side-channel attacks
    
    TO didefinisikan sebagai ukuran ketidakseimbangan dalam DDT
    Dihitung menggunakan varian dari formula:
    TO = Σ_{Δx≠0} max_{Δy} |{x : S(x) ⊕ S(x ⊕ Δx) = Δy}|
    
    Nilai yang lebih rendah = lebih baik (lebih uniform)
    """
    transparency_sum = 0
    
    for dx in range(1, 256):  # dx ≠ 0
        dy_counts = np.bincount(s ^ s[np.arange(256) ^ dx], minlength=256)
        transparency_sum += np.max(dy_counts)
    
    # Normalisasi
    to_value = transparency_sum / (255 * 256)
    return float(to_value)


def correlation_immunity(s, max_order=3):
    """
    Correlation Immunity (CI)
    Mengukur independensi output function dari subset input variables
    
    S-box bersifat correlation immune order k jika:
    - Output tidak berkorelasi dengan kombinasi ≤ k input bits
    
    Kita menghitung CI untuk setiap output bit dan mengembalikan minimum order
    
    Nilai yang lebih tinggi = lebih baik (lebih tahan correlation attacks)
    Untuk S-box 8-bit praktis: CI biasanya 0-3
    
    Note: Perhitungan penuh CI sangat mahal komputasi,
    ini adalah versi praktis dengan max_order limit
    """
    min_ci_order = max_order
    
    for bit_pos in range(8):
        # Ekstrak bit output tertentu
        f = np.array([1 - 2*((s[x] >> bit_pos) & 1) for x in range(256)], dtype=np.int32)
        
        # Walsh-Hadamard transform
        W = fwht(f.copy())
        
        ci_order = 0
        # Cek order dari 1 hingga max_order
        for order in range(1, max_order + 1):
            # Cek semua kombinasi dengan Hamming weight = order
            immune_at_order = True
            for idx in range(256):
                if bin(idx).count('1') == order:
                    if W[idx] != 0:
                        immune_at_order = False
                        break
            
            if immune_at_order:
                ci_order = order
            else:
                break
        
        min_ci_order = min(min_ci_order, ci_order)
    
    return int(min_ci_order)


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    sboxes = generate_sboxes(include_random=True, seed=42)
    results = []

    print("\n" + "="*70)
    print("PENGUJIAN S-BOX DENGAN METRIK TAMBAHAN (DU, AD, TO, CI)")
    print("="*70)

    for name, s in sboxes.items():
        print(f"\nMenghitung metrik untuk S-box: {name}...")
        
        res = {
            "S-box": name,
            "Balance": balance(s),
            "Bijective": bijective(s),
            "NL": nonlinearity(s),
            "SAC": sac(s),
            "LAP": lap(s),
            "DAP": dap(s),
            "BIC-SAC": bic_sac_fast(s),
            "BIC-NL": float(bic_nl_fast(s)),
            # Metrik baru
            "DU": differential_uniformity(s),
            "AD": algebraic_degree(s),
            "TO": transparency_order(s),
            "CI": correlation_immunity(s, max_order=3)
        }

        results.append(res)

        print(f"\n{'='*60}")
        print(f"HASIL UJI S-BOX: {name}")
        print(f"{'='*60}")
        print(f"{'Metrik':<15} {'Nilai':<20} {'Keterangan'}")
        print(f"{'-'*60}")
        
        # Metrik dasar
        print(f"{'Balance':<15} {str(res['Balance']):<20} {'Semua bit balanced' if res['Balance'] else 'Tidak balanced'}")
        print(f"{'Bijective':<15} {str(res['Bijective']):<20} {'Permutasi valid' if res['Bijective'] else 'Bukan permutasi'}")
        print(f"{'NL':<15} {res['NL']:<20} {'Nonlinearity (>100 baik)'}")
        print(f"{'SAC':<15} {res['SAC']:.6f}{'':>14} {'~0.5 ideal'}")
        print(f"{'LAP':<15} {res['LAP']:.6f}{'':>14} {'Mendekati 0 baik'}")
        print(f"{'DAP':<15} {res['DAP']:.6f}{'':>14} {'Mendekati 0 baik'}")
        print(f"{'BIC-SAC':<15} {res['BIC-SAC']:.6f}{'':>14} {'~0.5 ideal'}")
        print(f"{'BIC-NL':<15} {res['BIC-NL']:<20} {'Nonlinearity BIC'}")
        
        # Metrik baru
        print(f"\n{'METRIK TAMBAHAN':^60}")
        print(f"{'-'*60}")
        print(f"{'DU':<15} {res['DU']:<20} {'≤4 sangat baik (AES=4)'}")
        print(f"{'AD':<15} {res['AD']:<20} {'=7 optimal untuk 8-bit'}")
        print(f"{'TO':<15} {res['TO']:.6f}{'':>14} {'Lebih rendah lebih baik'}")
        print(f"{'CI':<15} {res['CI']:<20} {'Order immunity (0-3)'}")

    # Simpan hasil ke JSON
    with open("all_sbox_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "="*70)
    print("✅ Hasil disimpan ke all_sbox_results.json")
    
    # Simpan hasil ke CSV menggunakan pandas untuk format yang lebih baik
    import pandas as pd
    csv_filename = "all_sbox_results.csv"
    excel_filename = "all_sbox_results.xlsx"
    
    if results:
        df = pd.DataFrame(results)
        
        # Format numeric columns untuk readability
        numeric_cols = ['SAC', 'LAP', 'DAP', 'BIC-SAC', 'TO']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].round(8)
        
        # Export CSV
        df.to_csv(csv_filename, index=False)
        print(f"✅ Hasil disimpan ke {csv_filename}")
        
        # Export Excel dengan formatting
        try:
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='S-Box Metrics', index=False)
                
                # Auto-adjust column width
                worksheet = writer.sheets['S-Box Metrics']
                for idx, col in enumerate(df.columns):
                    max_length = max(
                        df[col].astype(str).apply(len).max(),
                        len(col)
                    ) + 2
                    worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 20)
            
            print(f"✅ Hasil disimpan ke {excel_filename}")
        except ImportError:
            print(f"⚠️  openpyxl tidak tersedia, skip Excel export")
        
        # Preview CSV
        print("\n📊 Preview Data (5 baris pertama):")
        print(df.head().to_string(index=False))
    
    print("="*70)
    
    # Tabel perbandingan lengkap
    print("\n" + "="*120)
    print("TABEL PERBANDINGAN S-BOX - SEMUA METRIK")
    print("="*120)
    print(f"{'S-Box':<10} {'Bal':<5} {'Bij':<5} {'NL':<5} {'SAC':<8} {'LAP':<8} {'DAP':<8} "
          f"{'BIC-SAC':<8} {'BIC-NL':<7} {'DU':<4} {'AD':<4} {'TO':<8} {'CI':<4}")
    print("-"*120)
    for res in results:
        bal = 'T' if res['Balance'] else 'F'
        bij = 'T' if res['Bijective'] else 'F'
        print(f"{res['S-box']:<10} {bal:<5} {bij:<5} {res['NL']:<5} {res['SAC']:<8.4f} "
              f"{res['LAP']:<8.6f} {res['DAP']:<8.6f} {res['BIC-SAC']:<8.4f} "
              f"{res['BIC-NL']:<7.0f} {res['DU']:<4} {res['AD']:<4} {res['TO']:<8.4f} {res['CI']:<4}")
    print("="*120)