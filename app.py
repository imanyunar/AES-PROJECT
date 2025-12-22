import warnings
import streamlit as st
import io
import csv
import numpy as np
import random
import hashlib
import json
import pandas as pd
from pathlib import Path
import logging
import time
# Suppress WebSocket warnings
logging.getLogger('tornado.access').setLevel(logging.ERROR)
logging.getLogger('tornado.application').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)


from sbox_utils import (
    generate_sboxes,
    balance,
    bijective,
    nonlinearity,
    sac,
    lap,
    dap,
    bic_sac_fast,
    bic_nl_fast 
)


from crypto import AESCustom

from crypto_functions import image_comparison_page
from image_encrypt_separate import image_encrypt_ui_new

from sbox_registry import extend_sbox_registry

# Page configuration
st.set_page_config(
    page_title="AES Encryption Suite",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Global Fade-in Animation */
    .stApp {
        animation: fadeIn 1.2s ease-in-out;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-image: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Tombol Modern */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        padding: 10px;
        transition: all 0.3s ease;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(37, 117, 252, 0.5);
    }

    /* Card untuk Metrik */
    [data-testid="stMetricValue"] {
        color: #00f2fe;
        font-family: 'Courier New', monospace;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS with animations
st.markdown("""
<style>
    /* Main animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 5px rgba(102, 126, 234, 0.5);
        }
        50% {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.8);
        }
    }
    
    /* Metric cards with gradient */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        color: white;
        animation: fadeInUp 0.5s ease-out;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        animation: pulse 2s infinite;
    }
    
    /* Success box with animation */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        color: #155724;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        animation: slideInRight 0.5s ease-out;
        box-shadow: 0 2px 10px rgba(40, 167, 69, 0.2);
    }
    
    /* Error box */
    .error-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
        color: #721c24;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        animation: slideInRight 0.5s ease-out;
        box-shadow: 0 2px 10px rgba(220, 53, 69, 0.2);
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 2px solid #17a2b8;
        color: #0c5460;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        animation: fadeInUp 0.5s ease-out;
        box-shadow: 0 2px 10px rgba(23, 162, 184, 0.2);
    }
    
    /* Button hover effect */
    .stButton>button {
        transition: all 0.3s ease;
        border-radius: 10px;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Selectbox animation */
    .stSelectbox {
        animation: fadeInUp 0.5s ease-out;
    }
    
    /* Headers with gradient text */
    h1, h2, h3 {
        animation: fadeInUp 0.7s ease-out;
    }
    
    /* Dataframe styling */
    .dataframe {
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        animation: slideInRight 0.5s ease-out;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f0f2f6 0%, #e9ecef 100%);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        transform: translateX(5px);
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Text input glow effect */
    .stTextInput>div>div>input:focus {
        animation: glow 2s infinite;
        border-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Path untuk file hasil metrics
METRICS_FILE = Path(__file__).parent / 'sbox_results.json'

# Initialize session state
if 'metrics_cache' not in st.session_state:
    st.session_state.metrics_cache = {}
if 'metrics_by_name' not in st.session_state:
    st.session_state.metrics_by_name = {}
if 'encrypted_result' not in st.session_state:
    st.session_state.encrypted_result = ''
if 'encrypted_result_decimal' not in st.session_state:
    st.session_state.encrypted_result_decimal = ''
if 'encrypted_result_text' not in st.session_state:
    st.session_state.encrypted_result_text = ''
if 'decrypted_result' not in st.session_state:
    st.session_state.decrypted_result = ''
if 'decrypted_result_decimal' not in st.session_state:
    st.session_state.decrypted_result_decimal = ''
if 'decrypted_result_text' not in st.session_state:
    st.session_state.decrypted_result_text = ''
if 'encrypted_image' not in st.session_state:
    st.session_state.encrypted_image = None
if 'decrypted_image' not in st.session_state:
    st.session_state.decrypted_image = None
if 'show_encrypted_image' not in st.session_state:
    st.session_state.show_encrypted_image = False
if 'show_decrypted_image' not in st.session_state:
    st.session_state.show_decrypted_image = False
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'stored_image_key' not in st.session_state:
    st.session_state.stored_image_key = ''
if 'image_sbox_name' not in st.session_state:
    st.session_state.image_sbox_name = ''
if 'show_sbox_metrics' not in st.session_state:
    st.session_state.show_sbox_metrics = False
if 'current_original_image' not in st.session_state:
    st.session_state.current_original_image = None
if 'decrypt_source_image' not in st.session_state:
    st.session_state.decrypt_source_image = None
if 'has_original_for_validation' not in st.session_state:
    st.session_state.has_original_for_validation = False
if 'decrypt_from_current_session' not in st.session_state:
    st.session_state.decrypt_from_current_session = True


def load_metrics_from_file():
    """Load pre-calculated metrics dari file JSON"""
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE, 'r') as f:
                data = json.load(f)
                
                st.session_state.metrics_by_name = data
                
                # Build hash-based cache
                sboxes = generate_sboxes(include_random=False)
                for name, sbox in sboxes.items():
                    if name in st.session_state.metrics_by_name:
                        sbox_hash = get_sbox_hash(sbox)
                        metrics = {k: v for k, v in st.session_state.metrics_by_name[name].items() if k != "S-box"}
                        st.session_state.metrics_cache[sbox_hash] = metrics
                
                return True
        except Exception as e:
            st.error(f"Error loading metrics file: {e}")
            st.session_state.metrics_cache = {}
            st.session_state.metrics_by_name = {}
            return False
    else:
        st.session_state.metrics_cache = {}
        st.session_state.metrics_by_name = {}
        return False

def save_metrics_to_file():
    """Save calculated metrics ke file JSON"""
    try:
        data = {}
        for name, metrics in st.session_state.metrics_by_name.items():
            data[name] = {
                "S-box": name,
                **metrics
            }
        
        with open(METRICS_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving metrics file: {e}")
        return False

# --- FUNGSI OPTIMASI: Tambahkan di bagian atas app.py ---

@st.cache_data
def load_all_metrics_json():
    """Membaca file JSON hasil pre-kalkulasi"""
    import json
    from pathlib import Path
    path = Path("sbox_results.json")
    if path.exists():
        with open(path, 'r') as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}

@st.cache_data
def calculate_and_cache_all_metrics(sbox_array, name):
    """Fallback: Hitung manual 8 metrik jika tidak ada di JSON"""
    from sbox_utils import (
        balance, bijective, nonlinearity, sac, 
        bic_sac_fast, bic_nl_fast, lap, dap
    )
    
    # Pastikan sbox dalam bentuk flat 256 elemen
    s_flat = sbox_array.flatten() if hasattr(sbox_array, 'flatten') else np.array(sbox_array).flatten()
    
    return {
        "Balance": balance(s_flat),
        "Bijective": bijective(s_flat),
        "Nonlinearity": nonlinearity(s_flat),
        "SAC": sac(s_flat),
        "BIC-SAC": bic_sac_fast(s_flat),
        "BIC-NL": bic_nl_fast(s_flat),
        "LAP": lap(s_flat),
        "DAP": dap(s_flat)
    }

def get_sbox_metrics_smart(sbox_name, sbox_array):
    """Fungsi Utama: Cek JSON dulu, baru hitung manual"""
    all_data_json = load_all_metrics_json()
    
    # Bersihkan nama dari emoji agar cocok dengan kunci di JSON
    clean_name = sbox_name.replace('📝 ', '').replace('🖼️ ', '').strip()
    
    # 1. Jika ada di JSON, ambil (Instan)
    if clean_name in all_data_json:
        return all_data_json[clean_name]
        
    # 2. Jika tidak ada, hitung manual (Akan lambat 1x saja, lalu masuk cache)
    return calculate_and_cache_all_metrics(sbox_array, clean_name)
def get_sbox_hash(sbox):
    """Generate unique hash untuk S-box"""
    return hashlib.md5(sbox.tobytes()).hexdigest()

def sbox_metrics(sbox, sbox_name=None, force_recalculate=False):
    """Hitung metrics S-box dengan caching"""
    if not force_recalculate and sbox_name and sbox_name in st.session_state.metrics_by_name:
        return st.session_state.metrics_by_name[sbox_name], True
    
    sbox_hash = get_sbox_hash(sbox)
    if not force_recalculate and sbox_hash in st.session_state.metrics_cache:
        return st.session_state.metrics_cache[sbox_hash], True
    
    metrics = {
        "Balance": balance(sbox),
        "Bijective": bijective(sbox),
        "NL": nonlinearity(sbox),
        "SAC": sac(sbox),
        "LAP": lap(sbox),
        "DAP": dap(sbox),
        "BIC-SAC": bic_sac_fast(sbox),
        "BIC-NL": bic_nl_fast(sbox)
    }
    
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            metrics_serializable[key] = float(value)
        elif isinstance(value, np.bool_):
            metrics_serializable[key] = bool(value)
        else:
            metrics_serializable[key] = value
    
    st.session_state.metrics_cache[sbox_hash] = metrics_serializable
    if sbox_name:
        st.session_state.metrics_by_name[sbox_name] = metrics_serializable
    
    return metrics_serializable, False

def calculate_progress_width(key, value):
    """Hitung width progress bar untuk setiap metric"""
    try:
        if key in ["Balance", "Bijective"]:
            return 100 if value else 0
        elif key == "NL" or key == "BIC-NL":
            return round((float(value) / 112) * 100)
        elif key in ["SAC", "BIC-SAC"]:
            return round(float(value) * 100)
        elif key in ["LAP", "DAP"]:
            return round((1 - float(value)) * 100)
        else:
            return 50
    except:
        return 50

def get_metric_score(key, value):
    """Calculate normalized score for ranking (0-100)"""
    try:
        if key in ["Balance", "Bijective"]:
            return 100 if value else 0
        elif key == "NL" or key == "BIC-NL":
            return round((float(value) / 112) * 100)
        elif key in ["SAC", "BIC-SAC"]:
            return round(float(value) * 100)
        elif key in ["LAP", "DAP"]:
            return round((1 - float(value)) * 100)
        else:
            return 50
    except:
        return 0

def parse_user_bytes(user_input, length=16):
    """Convert user input to bytes safely, truncate/pad ke length"""
    if not user_input:
        return bytes([0]*length)
    
    if isinstance(user_input, bytes):
        user_input = user_input.decode('utf-8', errors='ignore')
    
    user_input = str(user_input).strip()
    user_input_no_space = user_input.replace(" ", "")
    
    try:
        if all(c in '0123456789abcdefABCDEF' for c in user_input_no_space):
            b = bytes.fromhex(user_input_no_space)
        else:
            b = user_input.encode('utf-8')
    except ValueError:
        b = user_input.encode('utf-8')
    
    if len(b) > length:
        b = b[:length]
    elif len(b) < length:
        b = b.ljust(length, b'\x00')
    
    return b

def parse_custom_sbox(custom_sbox_str):
    """Parse custom S-box string with improved error handling"""
    if not custom_sbox_str or not custom_sbox_str.strip():
        raise ValueError("Custom S-box string is empty")
    
    sbox_list = []
    parts = custom_sbox_str.split(',')
    
    for i, part in enumerate(parts):
        part = part.strip()
        
        if not part:
            continue
        
        try:
            val = int(part, 0)
        except ValueError:
            try:
                val = int(part)
            except ValueError:
                raise ValueError(f"Cannot parse value at position {i}: '{part}'")
        
        if not (0 <= val <= 255):
            raise ValueError(f"Value at position {i} is out of range (0-255): {val}")
        
        sbox_list.append(val)
    
    if len(sbox_list) != 256:
        raise ValueError(f"Custom S-box must have exactly 256 elements, got {len(sbox_list)}")
    
    if len(set(sbox_list)) != 256:
        raise ValueError("Custom S-box must contain 256 unique values (bijective requirement)")
    
    return np.array(sbox_list, dtype=np.uint8)

def calculate_std_metrics(sboxes_dict=None):
    """Calculate standard deviation and other statistics for each metric across all S-boxes"""
    metrics_by_type = {}
    
    if sboxes_dict is None:
        data_source = st.session_state.metrics_by_name
    else:
        data_source = {}
        for name, sbox in sboxes_dict.items():
            metrics, _ = sbox_metrics(sbox, sbox_name=name, force_recalculate=False)
            data_source[name] = metrics
    
    for name, metrics in data_source.items():
        for metric_name, value in metrics.items():
            if metric_name not in ["Balance", "Bijective", "S-box"]:
                if metric_name not in metrics_by_type:
                    metrics_by_type[metric_name] = []
                metrics_by_type[metric_name].append(float(value))
    
    std_results = {}
    for metric_name, values in metrics_by_type.items():
        if len(values) > 1:
            std_results[metric_name] = {
                'std': float(np.std(values, ddof=1)),
                'mean': float(np.mean(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        else:
            std_results[metric_name] = {
                'std': 0.0,
                'mean': float(values[0]) if values else 0.0,
                'min': float(values[0]) if values else 0.0,
                'max': float(values[0]) if values else 0.0,
                'median': float(values[0]) if values else 0.0
            }
    
    return std_results

def display_metrics(metrics, from_cache=False):
    """Fungsi standar untuk menampilkan progress bar (Versi Lama)"""
    if from_cache:
        st.info("📦 Menggunakan metrik dari cache")
    
    for key, value in metrics.items():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write(f"**{key}:**")
        with col2:
            if key in ["Balance", "Bijective"]:
                st.write("✅ Pass" if value else "❌ Fail")
            else:
                # Menghitung lebar progress bar
                progress = 0
                try:
                    if "NL" in key or "Nonlinearity" in key: progress = float(value) / 112
                    elif "SAC" in key: progress = float(value)
                    elif "LAP" in key or "DAP" in key: progress = 1 - float(value)
                except: progress = 0.5
                
                st.progress(min(max(progress, 0.0), 1.0))
                st.write(f"{value}")

def display_metrics_8_lengkap(res):
    """Fungsi baru untuk menampilkan 8 metrik secara mendetail (Versi Upgrade)"""
    st.markdown("#### 📊 Ringkasan Metrik S-Box")
    m1, m2, m3, m4 = st.columns(4)
    # Mengambil data dengan fallback jika nama kunci berbeda
    nl_val = res.get("Nonlinearity") or res.get("NL", "N/A")
    sac_val = res.get("SAC", 0)
    bic_nl = res.get("BIC-NL", "N/A")
    bic_sac = res.get("BIC-SAC", 0)

    m1.metric("Nonlinearity", nl_val)
    m2.metric("SAC", f"{sac_val:.4f}")
    m3.metric("BIC-NL", bic_nl)
    m4.metric("BIC-SAC", f"{bic_sac:.4f}")

    with st.expander("🔍 Detail Hasil Pengujian Lengkap"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**⚖️ Balance State:** {'✅ Pass' if res.get('Balance') else '❌ Fail'}")
            st.write(f"**🔄 Bijective State:** {'✅ Pass' if res.get('Bijective') else '❌ Fail'}")
            st.write(f"**📈 Nonlinearity:** {nl_val}")
            st.write(f"**🎯 SAC:** {sac_val:.6f}")
        with col_b:
            st.write(f"**🧬 BIC-SAC:** {bic_sac:.6f}")
            st.write(f"**🧩 BIC-NL:** {bic_nl}")
            st.write(f"**📉 LAP:** {res.get('LAP', 0):.6f}")
            st.write(f"**📊 DAP:** {res.get('DAP', 0):.6f}")

# Load metrics on startup
if 'metrics_loaded' not in st.session_state:
    with st.spinner("Loading metrics..."):
        load_metrics_from_file()
        st.session_state.metrics_loaded = True

# Main title
st.title("🔐 AES Encryption Suite")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select Page", [
        "🏠 Home - Encryption/Decryption",
        "📊 S-Box Comparison",
        "🖼️ Image S-Box Comparison",
        "🔬 S-Box Testing",
        "📈 Statistics",
        "🔍 S-Box Viewer"
    ])
    
    st.markdown("---")
    st.info("**Tip:** Use the navigation above to explore different features of the AES Encryption Suite.")

# Page: Home - Encryption/Decryption
if page == "🏠 Home - Encryption/Decryption":
    st.header("Encryption & Decryption")
    
    # S-box selection dengan kategorisasi
    from image_crypto import test_image_sboxes
    
    sboxes = generate_sboxes(include_random=True)
    # Tambahkan image sboxes dengan prefix "IMG-"
    image_sboxes_dict = test_image_sboxes()
    for name, sbox in image_sboxes_dict.items():
        sboxes[f"IMG-{name}"] = sbox
    
    # Kategorisasi S-boxes
    text_sboxes = sorted([name for name in sboxes.keys() if not name.startswith("IMG-") and name not in ["RANDOM"]])
    image_sboxes = sorted([name for name in sboxes.keys() if name.startswith("IMG-")])
    
    # Gabungkan dalam urutan yang rapi
    all_sboxes = text_sboxes + image_sboxes + ["RANDOM", "CUSTOM"]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        sbox_choice = st.selectbox(
            "Select S-Box", 
            all_sboxes, 
            index=0,
            help="📝 Text S-boxes (AES, A0-A2, K4/K44/K128) | 🖼️ Image S-boxes (IMG-S-box1/2/3) | 🎲 Dynamic (RANDOM/CUSTOM)"
        )
        
        # Display S-box type info dengan styling yang lebih baik
        if sbox_choice.startswith("IMG-"):
            st.info("🖼️ **Image S-box** - Optimized for image encryption")
        elif sbox_choice == "RANDOM":
            st.warning("🎲 **Random S-box** - Click 'Generate Random S-Box' button below")
        elif sbox_choice == "CUSTOM":
            st.warning("✏️ **Custom S-box** - Enter your own 256 values below")
        elif sbox_choice == "AES":
            st.success("🔒 **AES Standard S-box** - Rijndael's original design used in AES-128/192/256")
        else:
            st.info("📝 **Text S-box** - Generated using affine matrix transformation")
            
    with col2:
        # Show quick stats when S-box is selected
        if sbox_choice and sbox_choice in sboxes:
            st.markdown("#### 📊 Quick Info")
            sbox_type = "🖼️ Image" if sbox_choice.startswith("IMG-") else "📝 Text"
            st.caption(f"**Type:** {sbox_type}")
            st.caption(f"**Size:** 256 bytes")
            st.caption(f"**Name:** {sbox_choice}")
        
        if st.button("🔄 Recalculate Metrics", key="recalc_single", use_container_width=True):
            if sbox_choice in sboxes:
                with st.spinner("Recalculating..."):
                    active_sbox = sboxes[sbox_choice]
                    metrics, _ = sbox_metrics(active_sbox, sbox_name=sbox_choice, force_recalculate=True)
                    st.success("✅ Metrics recalculated!")
    
    # Custom S-box input (sisanya tetap sama)
    custom_sbox_str = ""
    if sbox_choice == "CUSTOM":
        st.markdown("---")
        st.markdown("### ✏️ Custom S-Box Input")
        custom_sbox_str = st.text_area(
            "Enter 256 comma-separated values (0-255)",
            height=100,
            placeholder="0,1,2,3,4,5,...,254,255",
            help="Must be a valid bijective mapping - all values 0-255 must appear exactly once"
        )
        
        # Show example
        with st.expander("📖 See Example"):
            st.code("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,...,255")
            st.caption("💡 Tip: Use Excel or Python to generate and validate your S-box")
    
    # Determine active S-box (sisanya tetap sama)
    active_sbox = None
    if sbox_choice in sboxes:
        active_sbox = sboxes[sbox_choice]
    elif sbox_choice == "RANDOM":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎲 Generate Random S-Box", use_container_width=True):
                active_sbox = np.array(random.sample(range(256), 256), dtype=np.uint8)
                st.session_state.random_sbox = active_sbox
                st.success("✅ Random S-box generated!")
                st.balloons()
        with col2:
            if 'random_sbox' in st.session_state:
                if st.button("🔄 Regenerate", use_container_width=True):
                    active_sbox = np.array(random.sample(range(256), 256), dtype=np.uint8)
                    st.session_state.random_sbox = active_sbox
                    st.success("✅ Regenerated!")
        
        if 'random_sbox' in st.session_state:
            active_sbox = st.session_state.random_sbox
            st.info("✅ Random S-box is active and ready to use")
    elif sbox_choice == "CUSTOM" and custom_sbox_str:
        try:
            active_sbox = parse_custom_sbox(custom_sbox_str)
            st.success("✅ Custom S-box parsed and validated successfully!")
            st.info(f"🎯 All 256 unique values confirmed - S-box is bijective")
        except Exception as e:
            st.error(f"❌ Error parsing custom S-box: {str(e)}")
            st.info("💡 Make sure you have exactly 256 comma-separated values (0-255) with no duplicates")
            active_sbox = None
    
    # Display S-box metrics - OPTIONAL (tidak otomatis)
    if active_sbox is not None and sbox_choice not in ["RANDOM", "CUSTOM"]:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### 📊 S-Box Selected: **{sbox_choice}**")
        with col2:
            if st.button("📈 Show Metrics", use_container_width=True, key="show_metrics_btn"):
                st.session_state.show_sbox_metrics = True
        
        if st.session_state.get('show_sbox_metrics', False):
            try:
                with st.spinner("Calculating metrics..."):
                    metrics, from_cache = sbox_metrics(active_sbox, sbox_name=sbox_choice, force_recalculate=False)
                st.success("✅ Metrics loaded")
                display_metrics(metrics, from_cache)
                
                if st.button("🗑️ Hide Metrics", key="hide_metrics_btn"):
                    st.session_state.show_sbox_metrics = False
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error calculating metrics: {str(e)}")
    
    # Download S-box
    if active_sbox is not None:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬇️ Download S-Box CSV", use_container_width=True):
                csv_buffer = io.StringIO()
                writer = csv.writer(csv_buffer)
                for val in active_sbox:
                    writer.writerow([val])
                
                st.download_button(
                    label="Save CSV File",
                    data=csv_buffer.getvalue(),
                    file_name=f"{sbox_choice}_sbox.csv",
                    mime="text/csv",
                    key="download_sbox_csv"
                )
    
    st.markdown("---")
    
    # Encryption/Decryption
    st.subheader("🔐 Text Encryption/Decryption")
    
    col1, col2 = st.columns(2)
    with col1:
        plaintext_input = st.text_input("Input Text (max 16 bytes)", value="", key="text_input_field")
    with col2:
        key_input = st.text_input("Key (16 bytes)", value="", type="password", key="text_key_field")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔒 Encrypt", use_container_width=True, key="encrypt_text_btn"):
            if active_sbox is not None:
                try:
                    pt_bytes = parse_user_bytes(plaintext_input, length=16)
                    key_bytes = parse_user_bytes(key_input, length=16)
                    
                    aes = AESCustom(sbox_name=sbox_choice, key_bytes=key_bytes)
                    aes.sbox = active_sbox
                    
                    aes.inv_sbox = np.zeros(256, dtype=np.uint8)
                    for i in range(256):
                        aes.inv_sbox[aes.sbox[i]] = i
                    
                    ct_bytes = aes.encrypt(pt_bytes)
                    st.session_state.encrypted_result = ct_bytes.hex()
                    st.session_state.encrypted_result_decimal = ' '.join(str(b) for b in ct_bytes)
                    try:
                        st.session_state.encrypted_result_text = ct_bytes.decode('utf-8', errors='replace')
                    except:
                        st.session_state.encrypted_result_text = '(non-printable)'
                    
                    st.success("✅ Encryption completed!")
                    
                except Exception as e:
                    st.error(f"❌ Error during encryption: {str(e)}")
            else:
                st.error("❌ Please select or define an S-box first")
    
    with col2:
        if st.button("🔓 Decrypt", use_container_width=True, key="decrypt_text_btn"):
            if active_sbox is not None:
                try:
                    ct_bytes = parse_user_bytes(plaintext_input, length=16)
                    key_bytes = parse_user_bytes(key_input, length=16)
                    
                    aes = AESCustom(sbox_name=sbox_choice, key_bytes=key_bytes)
                    aes.sbox = active_sbox
                    
                    aes.inv_sbox = np.zeros(256, dtype=np.uint8)
                    for i in range(256):
                        aes.inv_sbox[aes.sbox[i]] = i
                    
                    pt_bytes = aes.decrypt(ct_bytes)
                    st.session_state.decrypted_result = pt_bytes.hex()
                    st.session_state.decrypted_result_decimal = ' '.join(str(b) for b in pt_bytes)
                    try:
                        st.session_state.decrypted_result_text = pt_bytes.decode('utf-8', errors='replace')
                    except:
                        st.session_state.decrypted_result_text = '(non-printable)'
                    
                    st.success("✅ Decryption completed!")
                    
                except Exception as e:
                    st.error(f"❌ Error during decryption: {str(e)}")
            else:
                st.error("❌ Please select or define an S-box first")
    
    # Display Encryption Results
    if st.session_state.encrypted_result:
        st.markdown("### 🔒 Encryption Results")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_input("Hex (Encrypted)", st.session_state.encrypted_result, disabled=True, key="enc_hex")
        with col2:
            if st.button("📋 Copy", key="copy_enc"):
                st.toast("Copied to clipboard!")
        
        st.text_input("Decimal (Encrypted)", st.session_state.encrypted_result_decimal, disabled=True, key="enc_dec")
        
        text_clean = st.session_state.encrypted_result_text.rstrip('\x00').replace('\x00', '')
        st.text_input("Text (Encrypted)", text_clean, disabled=True, key="enc_text")
        
        st.caption(f"📏 Length: {len(bytes.fromhex(st.session_state.encrypted_result))} bytes")
        
        if st.button("🗑️ Clear Encryption Result"):
            st.session_state.encrypted_result = ''
            st.session_state.encrypted_result_decimal = ''
            st.session_state.encrypted_result_text = ''
            st.rerun()
    
    # Display Decryption Results
    if st.session_state.decrypted_result:
        st.markdown("### 🔓 Decryption Results")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_input("Hex (Decrypted)", st.session_state.decrypted_result, disabled=True, key="dec_hex")
        with col2:
            if st.button("📋 Copy", key="copy_dec"):
                st.toast("Copied to clipboard!")
        
        st.text_input("Decimal (Decrypted)", st.session_state.decrypted_result_decimal, disabled=True, key="dec_dec")
        
        text_clean = st.session_state.decrypted_result_text.rstrip('\x00').replace('\x00', '')
        st.text_input("Text (Decrypted)", text_clean, disabled=True, key="dec_text")
        
        st.caption(f"📏 Length: {len(bytes.fromhex(st.session_state.decrypted_result))} bytes")
        
        if st.button("🗑️ Clear Decryption Result"):
            st.session_state.decrypted_result = ''
            st.session_state.decrypted_result_decimal = ''
            st.session_state.decrypted_result_text = ''
            st.rerun()

    # ==============================
    # IMAGE ENCRYPTION (TAMBAHAN)
    # ==============================
    image_encrypt_ui_new(active_sbox, sbox_choice)

    # Page: Image S-Box Comparison
elif page == "🖼️ Image S-Box Comparison":
    sboxes = generate_sboxes(include_random=False)
    sboxes = extend_sbox_registry(sboxes)
    image_comparison_page(sboxes)

# Page: S-Box Comparison
# In S-Box Comparison page
# Page: S-Box Comparison
elif page == "📊 S-Box Comparison":
    st.header("S-Box Comparison")
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📊 View Results", "🔄 Recalculate Metrics", "📈 Statistics"])
    
    # ============================================================================
    # TAB 1: VIEW RESULTS
    # ============================================================================
    with tab1:
        st.subheader("S-Box Comparison Results")
        
        # Generate all S-boxes
        sboxes = generate_sboxes(include_random=False)
        
        # Add image S-boxes
        try:
            from image_crypto import test_image_sboxes
            image_sboxes_dict = test_image_sboxes()
            for name, sbox in image_sboxes_dict.items():
                sboxes[f"IMG-{name}"] = sbox
        except Exception as e:
            st.warning(f"⚠️ Could not load image S-boxes: {e}")
        
        # Check if metrics are available
        total_sboxes = len(sboxes)
        available_metrics = len(st.session_state.metrics_by_name)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total S-Boxes", total_sboxes)
        with col2:
            st.metric("Metrics Available", available_metrics)
        with col3:
            missing = total_sboxes - available_metrics
            if missing > 0:
                st.metric("Missing Metrics", missing, delta=f"-{missing}", delta_color="inverse")
            else:
                st.metric("Status", "✅ Complete")
        
        st.markdown("---")
        
        # Build comparison data
        comparison_data = {}
        missing_metrics = []
        
        for name, sbox in sboxes.items():
            try:
                metrics, from_cache = sbox_metrics(sbox, sbox_name=name, force_recalculate=False)
                
                if not metrics:
                    missing_metrics.append(name)
                    continue
                
                # Calculate overall score
                scores = []
                for key, value in metrics.items():
                    if key not in ["Balance", "Bijective"]:
                        scores.append(get_metric_score(key, value))
                
                overall_score = sum(scores) / len(scores) if scores else 0
                
                comparison_data[name] = {
                    'metrics': metrics,
                    'overall_score': round(overall_score, 2),
                    'from_cache': from_cache
                }
            except Exception as e:
                st.error(f"Error processing {name}: {e}")
                missing_metrics.append(name)
        
        # Show warning if missing metrics
        if missing_metrics:
            st.warning(f"⚠️ Missing metrics for: {', '.join(missing_metrics)}")
            st.info("💡 Go to 'Recalculate Metrics' tab to calculate missing S-boxes")
        
        # Display results if available
        if comparison_data:
            st.subheader("🔢 Detailed Comparison Table")
            
            # Build comparison dataframe
            comparison_rows = []
            for name, data in comparison_data.items():
                row = {
                    'S-Box': name,
                    'Overall Score': data['overall_score']
                }
                row.update(data['metrics'])
                comparison_rows.append(row)
            
            df = pd.DataFrame(comparison_rows)
            df = df.sort_values('Overall Score', ascending=False)
            
            # Format and display
            st.dataframe(df.style.format({
                'Overall Score': '{:.2f}',
                'NL': '{:.2f}',
                'SAC': '{:.4f}',
                'LAP': '{:.6f}',
                'DAP': '{:.6f}',
                'BIC-SAC': '{:.4f}',
                'BIC-NL': '{:.2f}'
            }).background_gradient(subset=['Overall Score'], cmap='RdYlGn'), 
            use_container_width=True)
            
            st.markdown("---")
            
            # Individual S-box comparison
            st.subheader("🔍 Individual S-Box Analysis")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_sboxes = st.multiselect(
                    "Select S-Boxes to Compare (2 or more recommended)",
                    options=list(sboxes.keys()),
                    default=list(sboxes.keys())[:3] if len(sboxes) >= 3 else list(sboxes.keys()),
                    help="Select multiple S-boxes to see detailed comparison"
                )
            
            with col2:
                st.write("")
                st.write("")
                if st.button("📊 Select All", use_column_width=True):
                    st.session_state.selected_all = True
                    st.rerun()
                
                if st.session_state.get('selected_all', False):
                    selected_sboxes = list(sboxes.keys())
                    st.session_state.selected_all = False
            
            if len(selected_sboxes) >= 1:
                st.markdown(f"**Comparing {len(selected_sboxes)} S-Box(es):**")
                
                # Prepare data for comparison
                comparison_detail = []
                for name in selected_sboxes:
                    if name in comparison_data:
                        comparison_detail.append({
                            'S-Box': name,
                            **comparison_data[name]['metrics'],
                            'Overall Score': comparison_data[name]['overall_score']
                        })
                
                if comparison_detail:
                    df_selected = pd.DataFrame(comparison_detail)
                    
                    # Display as styled table
                    st.dataframe(df_selected.style.format({
                        'Overall Score': '{:.2f}',
                        'NL': '{:.2f}',
                        'SAC': '{:.4f}',
                        'LAP': '{:.6f}',
                        'DAP': '{:.6f}',
                        'BIC-SAC': '{:.4f}',
                        'BIC-NL': '{:.2f}'
                    }).background_gradient(subset=['Overall Score'], cmap='RdYlGn'), 
                    use_container_width=True)
                    
                    # Visualization for selected S-boxes
                    if len(selected_sboxes) >= 2:
                        st.markdown("---")
                        st.subheader("📊 Metric Comparison Charts")
                        
                        # Prepare data for charts (non-boolean metrics only)
                        chart_metrics = {
                            'NL': 'Nonlinearity (Higher is better)',
                            'SAC': 'Strict Avalanche Criterion (Closer to 0.5 is better)',
                            'LAP': 'Linear Approximation Probability (Lower is better)',
                            'DAP': 'Differential Approximation Probability (Lower is better)',
                            'BIC-SAC': 'BIC-SAC (Closer to 0.5 is better)',
                            'BIC-NL': 'BIC Nonlinearity (Higher is better)'
                        }
                        
                        # Create 2 columns for charts
                        col1, col2 = st.columns(2)
                        
                        for idx, (metric, description) in enumerate(chart_metrics.items()):
                            if metric in df_selected.columns:
                                with col1 if idx % 2 == 0 else col2:
                                    st.write(f"**{metric}** - {description}")
                                    chart_data = df_selected[['S-Box', metric]].set_index('S-Box')
                                    st.bar_chart(chart_data)
                    
                    # Download selected comparison
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv_buffer_selected = io.StringIO()
                        df_selected.to_csv(csv_buffer_selected, index=False)
                        st.download_button(
                            label=f"⬇️ Download Selected Comparison ({len(selected_sboxes)} S-Boxes)",
                            data=csv_buffer_selected.getvalue(),
                            file_name=f"sbox_comparison_selected_{len(selected_sboxes)}.csv",
                            mime="text/csv",
                            key="download_selected",
                            use_column_width=True
                        )
                    
                    with col2:
                        # Download as JSON
                        json_data = df_selected.to_dict(orient='records')
                        st.download_button(
                            label=f"📋 Download as JSON",
                            data=json.dumps(json_data, indent=4),
                            file_name=f"sbox_comparison_selected_{len(selected_sboxes)}.json",
                            mime="application/json",
                            key="download_json_selected",
                            use_column_width=True
                        )
            
            # Download all comparison
            st.markdown("---")
            st.subheader("📥 Download All Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label=f"📄 Download All as CSV ({len(df)} S-Boxes)",
                    data=csv_buffer.getvalue(),
                    file_name="sbox_comparison_all.csv",
                    mime="text/csv",
                    key="download_all_csv",
                    use_column_width=True
                )
            
            with col2:
                json_buffer = json.dumps(comparison_rows, indent=4)
                st.download_button(
                    label=f"📋 Download All as JSON",
                    data=json_buffer,
                    file_name="sbox_comparison_all.json",
                    mime="application/json",
                    key="download_all_json",
                    use_column_width=True
                )
            
            with col3:
                # Create Excel-like format
                excel_data = df.to_csv(sep='\t', index=False)
                st.download_button(
                    label=f"📊 Download as TSV (Excel)",
                    data=excel_data,
                    file_name="sbox_comparison_all.tsv",
                    mime="text/tab-separated-values",
                    key="download_all_tsv",
                    use_column_width=True
                )
        
        else:
            st.warning("⚠️ No metrics available.")
            st.info("💡 Please go to the 'Recalculate Metrics' tab to calculate S-box metrics first.")
    
    # ============================================================================
    # TAB 2: RECALCULATE METRICS
    # ============================================================================
    with tab2:
        st.subheader("🔄 Recalculate S-Box Metrics")
        
        st.info("💡 Choose a recalculation mode based on your needs. Single S-Box mode is fastest and prevents timeout.")
        
        # Recalculation mode selector
        recalc_mode = st.selectbox(
            "Select Recalculation Mode",
            ["Single S-Box", "Batch (3 at once)", "All at once"],
            help="Choose calculation mode to prevent timeout"
        )
        
        st.markdown("---")
        
        # === SINGLE S-BOX RECALCULATION ===
        if recalc_mode == "Single S-Box":
            st.markdown("### 🎯 Single S-Box Mode")
            st.write("Calculate metrics for one S-Box at a time (~8-10 seconds per S-box)")
            
            sboxes = generate_sboxes(include_random=False)
            
            # Add image S-boxes
            try:
                from image_crypto import test_image_sboxes
                image_sboxes_dict = test_image_sboxes()
                for name, sbox in image_sboxes_dict.items():
                    sboxes[f"IMG-{name}"] = sbox
            except:
                pass
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_sbox = st.selectbox(
                    "Select S-Box to recalculate",
                    list(sboxes.keys()),
                    help="Choose which S-Box to recalculate"
                )
            
            with col2:
                st.write("")
                st.write("")
                # Show status
                if selected_sbox in st.session_state.metrics_by_name:
                    st.success("✅ Has metrics")
                else:
                    st.warning("⚠️ No metrics")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                if st.button("🔄 Recalculate Selected", key="recalc_single_btn", use_column_width=True):
                    with st.spinner(f"⏳ Calculating metrics for {selected_sbox}..."):
                        sbox = sboxes[selected_sbox]
                        
                        # Progress indicator
                        progress_text = st.empty()
                        progress_bar = st.progress(0)
                        
                        progress_text.text("📊 Running 8 cryptographic tests...")
                        progress_bar.progress(0.1)
                        
                        start_time = time.time()
                        metrics, _ = sbox_metrics(sbox, sbox_name=selected_sbox, force_recalculate=True)
                        elapsed = time.time() - start_time
                        
                        progress_bar.progress(0.9)
                        save_metrics_to_file()
                        progress_bar.progress(1.0)
                        
                        progress_text.empty()
                        progress_bar.empty()
                        
                        st.success(f"✅ Recalculated {selected_sbox} in {elapsed:.1f} seconds")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
            
            with col2:
                # Recalculate all text S-boxes
                if st.button("🔄 Recalculate All Text S-Boxes", key="recalc_text_btn", use_column_width=True):
                    text_sboxes = {k: v for k, v in sboxes.items() if not k.startswith("IMG-")}
                    st.info(f"⏳ Processing {len(text_sboxes)} text S-boxes...")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    time_text = st.empty()
                    
                    start_time = time.time()
                    
                    for idx, (name, sbox) in enumerate(text_sboxes.items()):
                        status_text.text(f"Processing: {name} ({idx+1}/{len(text_sboxes)})")
                        metrics, _ = sbox_metrics(sbox, sbox_name=name, force_recalculate=True)
                        progress_bar.progress((idx + 1) / len(text_sboxes))
                        
                        elapsed = time.time() - start_time
                        eta = (elapsed / (idx + 1)) * (len(text_sboxes) - idx - 1)
                        time_text.text(f"⏱️ Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
                    
                    save_metrics_to_file()
                    
                    total_time = time.time() - start_time
                    status_text.empty()
                    time_text.empty()
                    
                    st.success(f"✅ Recalculated {len(text_sboxes)} text S-boxes in {total_time:.1f}s!")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
            
            with col3:
                st.caption("💡 Safe mode prevents timeout")
        
        # === BATCH RECALCULATION ===
        elif recalc_mode == "Batch (3 at once)":
            st.markdown("### 📦 Batch Processing Mode")
            st.write("Process 3 S-boxes at a time to prevent timeout (~25-30 seconds per batch)")
            
            sboxes = generate_sboxes(include_random=False)
            
            # Add image S-boxes
            try:
                from image_crypto import test_image_sboxes
                image_sboxes_dict = test_image_sboxes()
                for name, sbox in image_sboxes_dict.items():
                    sboxes[f"IMG-{name}"] = sbox
            except:
                pass
            
            sbox_list = list(sboxes.items())
            
            # Calculate batches
            batch_size = 3
            total_batches = (len(sbox_list) + batch_size - 1) // batch_size
            
            if 'current_batch' not in st.session_state:
                st.session_state.current_batch = 0
            
            # Show progress
            progress = st.session_state.current_batch / total_batches if total_batches > 0 else 0
            st.progress(progress)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Batch", f"{st.session_state.current_batch}/{total_batches}")
            with col2:
                completed = st.session_state.current_batch * batch_size
                st.metric("Completed", f"{min(completed, len(sbox_list))}/{len(sbox_list)}")
            with col3:
                remaining = len(sbox_list) - completed
                st.metric("Remaining", max(0, remaining))
            
            st.markdown("---")
            
            if st.session_state.current_batch < total_batches:
                # Get current batch
                start_idx = st.session_state.current_batch * batch_size
                end_idx = min(start_idx + batch_size, len(sbox_list))
                current_batch = dict(sbox_list[start_idx:end_idx])
                
                st.write(f"**Next batch ({len(current_batch)} S-boxes):** {', '.join(current_batch.keys())}")
                
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    if st.button("▶️ Process Next Batch", key="process_batch_btn", use_column_width=True):
                        with st.spinner(f"Processing batch {st.session_state.current_batch + 1}/{total_batches}..."):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for idx, (name, sbox) in enumerate(current_batch.items()):
                                status_text.text(f"Calculating: {name} ({idx+1}/{len(current_batch)})")
                                metrics, _ = sbox_metrics(sbox, sbox_name=name, force_recalculate=True)
                                progress_bar.progress((idx + 1) / len(current_batch))
                            
                            save_metrics_to_file()
                            st.session_state.current_batch += 1
                            
                            status_text.empty()
                            progress_bar.empty()
                            
                            st.success(f"✅ Batch {st.session_state.current_batch} complete!")
                            time.sleep(1)
                            st.rerun()
                
                with col2:
                    if st.button("⏭️ Skip to Results", key="skip_batch_btn", use_column_width=True):
                        st.session_state.current_batch = total_batches
                        st.info("Skipped to results view")
                        st.rerun()
                
                with col3:
                    st.caption("💡 Step by step processing")
            
            else:
                st.success("🎉 All batches processed!")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Reset and Restart", use_column_width=True):
                        st.session_state.current_batch = 0
                        st.info("Batch counter reset to 0")
                        st.rerun()
                
                with col2:
                    if st.button("📊 View Results", use_column_width=True):
                        st.info("Switch to 'View Results' tab to see comparison")
        
        # === ALL AT ONCE (Original behavior) ===
        else:
            st.markdown("### ⚡ All at Once Mode")
            st.warning("⚠️ **Warning:** This may take 60-120 seconds. Your browser may show 'not responding' - this is normal!")
            st.write("This mode calculates all S-boxes in one go. Use only if you need to recalculate everything.")
            
            # Add confirmation checkbox
            confirm_all = st.checkbox(
                "⚠️ I understand this will take time and may freeze the browser temporarily",
                key="confirm_all_recalc"
            )
            
            st.markdown("---")
            
            if st.button("🔄 Recalculate All S-Boxes", key="recalc_all_btn", disabled=not confirm_all, use_column_width=True):
                sboxes = generate_sboxes(include_random=False)
                
                # Add image S-boxes
                try:
                    from image_crypto import test_image_sboxes
                    image_sboxes_dict = test_image_sboxes()
                    for name, sbox in image_sboxes_dict.items():
                        sboxes[f"IMG-{name}"] = sbox
                except:
                    pass
                
                total = len(sboxes)
                
                st.info(f"⏳ Processing {total} S-boxes... Please wait and do not close this tab!")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_text = st.empty()
                
                overall_start = time.time()
                
                for idx, (name, sbox) in enumerate(sboxes.items()):
                    status_text.text(f"⏳ Calculating: {name} ({idx+1}/{total})")
                    
                    start = time.time()
                    metrics, _ = sbox_metrics(sbox, sbox_name=name, force_recalculate=True)
                    elapsed_single = time.time() - start
                    
                    progress_bar.progress((idx + 1) / total)
                    
                    elapsed_total = time.time() - overall_start
                    if idx > 0:
                        eta = (elapsed_total / (idx + 1)) * (total - idx - 1)
                        time_text.text(f"⏱️ Elapsed: {elapsed_total:.1f}s | Last: {elapsed_single:.1f}s | ETA: {eta:.1f}s")
                
                save_metrics_to_file()
                
                total_time = time.time() - overall_start
                
                status_text.empty()
                time_text.empty()
                progress_bar.empty()
                
                st.success(f"✅ Successfully recalculated {total} S-boxes in {total_time:.1f} seconds ({total_time/60:.1f} minutes)!")
                st.balloons()
                
                # Save timestamp
                import datetime
                st.session_state.last_recalc_time = datetime.datetime.now()
                
                time.sleep(2)
                st.rerun()
            
            if not confirm_all:
                st.info("👆 Please check the confirmation box above to enable the button")
    
    # ============================================================================
    # TAB 3: STATISTICS
    # ============================================================================
    with tab3:
        st.subheader("📈 Statistical Analysis")
        
        stats = calculate_std_metrics()
        
        if stats:
            st.write("Statistical summary of cryptographic metrics across all S-boxes:")
            
            # Display statistics table
            stats_df = pd.DataFrame(stats).T
            stats_df = stats_df[['mean', 'std', 'min', 'max', 'median']]  # Reorder columns
            
            st.dataframe(stats_df.style.format("{:.6f}").background_gradient(subset=['mean'], cmap='Blues'), 
                        use_container_width=True)
            
            st.markdown("---")
            
            # Visualizations
            st.subheader("📊 Statistical Visualizations")
            
            for metric_name, values in stats.items():
                with st.expander(f"📈 {metric_name} Statistics", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Mean", f"{values['mean']:.6f}")
                        st.metric("Standard Deviation", f"{values['std']:.6f}")
                    
                    with col2:
                        st.metric("Minimum", f"{values['min']:.6f}")
                        st.metric("Maximum", f"{values['max']:.6f}")
                    
                    with col3:
                        st.metric("Median", f"{values['median']:.6f}")
                        range_val = values['max'] - values['min']
                        st.metric("Range", f"{range_val:.6f}")
                    
                    # Create bar chart for min/mean/max
                    chart_data = pd.DataFrame({
                        'Statistic': ['Min', 'Mean', 'Median', 'Max'],
                        'Value': [values['min'], values['mean'], values['median'], values['max']]
                    })
                    st.bar_chart(chart_data.set_index('Statistic'))
            
            # Download statistics
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_buffer = io.StringIO()
                stats_df.to_csv(csv_buffer)
                st.download_button(
                    label="⬇️ Download Statistics as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="sbox_statistics.csv",
                    mime="text/csv",
                    key="download_stats_csv",
                    use_column_width=True
                )
            
            with col2:
                json_stats = json.dumps(stats, indent=4)
                st.download_button(
                    label="📋 Download Statistics as JSON",
                    data=json_stats,
                    file_name="sbox_statistics.json",
                    mime="application/json",
                    key="download_stats_json",
                    use_column_width=True
                )
        
        else:
            st.warning("⚠️ No statistics available.")
            st.info("💡 Please calculate S-box metrics first in the 'Recalculate Metrics' tab.")
# Page: S-Box Testing
# ================= Page: S-Box Testing =================
elif page == "🔬 S-Box Testing":
    st.header("🔬 S-Box Quality Testing")

    # Generate once only
    sboxes = generate_sboxes(include_random=False)

    sbox_name = st.selectbox(
        "Select S-Box for Testing",
        list(sboxes.keys())
    )

    if st.button("🧪 Run Detailed Test"):
        with st.spinner(f"Testing {sbox_name}..."):
            s = sboxes[sbox_name]

            result = {
                "S-box": sbox_name,
                "Balance": balance(s),
                "Bijective": bijective(s),
                "NL": nonlinearity(s),
                "SAC": sac(s),
                "LAP": lap(s),
                "DAP": dap(s),
                "BIC-SAC": bic_sac_fast(s),
                "BIC-NL": bic_nl_fast(s)
            }

        st.success("✅ Test completed")
        st.json(result)


# Page: Statistics
elif page == "📈 Statistics":
    st.header("Statistical Analysis")
    
    stats = calculate_std_metrics()
    
    if stats:
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📊 Summary Table", "📈 Visualizations"])
        
        with tab1:
            stats_df = pd.DataFrame(stats).T
            st.dataframe(stats_df.style.format("{:.6f}"), use_container_width=True)
            
            # Download statistics
            csv_buffer = io.StringIO()
            stats_df.to_csv(csv_buffer)
            st.download_button(
                label="⬇️ Download Statistics CSV",
                data=csv_buffer.getvalue(),
                file_name="sbox_statistics.csv",
                mime="text/csv"
            )
        
        with tab2:
            # Display charts for each metric
            for metric_name, values in stats.items():
                st.subheader(metric_name)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mean", f"{values['mean']:.6f}")
                with col2:
                    st.metric("Std Dev", f"{values['std']:.6f}")
                with col3:
                    st.metric("Median", f"{values['median']:.6f}")
                
                # Create bar chart for min/max
                chart_data = pd.DataFrame({
                    'Statistic': ['Min', 'Max', 'Mean'],
                    'Value': [values['min'], values['max'], values['mean']]
                })
                st.bar_chart(chart_data.set_index('Statistic'))
                st.markdown("---")
    else:
        st.warning("No statistics available. Please load or calculate metrics first.")

# Page: S-Box Viewer
elif page == "🔍 S-Box Viewer":
    st.header("S-Box Data Viewer")
    
    # Import image sboxes
    from image_crypto import test_image_sboxes
    
    # Combine text and image sboxes
    text_sboxes = generate_sboxes(include_random=False)
    image_sboxes = test_image_sboxes()
    
    # Create combined dictionary with prefixes
    all_sboxes = {}
    for name, sbox in text_sboxes.items():
        all_sboxes[f"📝 {name}"] = sbox
    for name, sbox in image_sboxes.items():
        all_sboxes[f"🖼️ {name}"] = sbox
    
    sbox_name = st.selectbox("Select S-Box", list(all_sboxes.keys()))
    
    sbox = all_sboxes[sbox_name]
    
    # Display format selection
    display_format = st.radio("Display Format", ["Hexadecimal", "Decimal"], horizontal=True)
    
    # Create 16x16 grid
    st.subheader(f"{sbox_name} S-Box ({display_format})")
    
    # Create dataframe for display
    if display_format == "Hexadecimal":
        data = [[f"{sbox[i*16 + j]:02X}" for j in range(16)] for i in range(16)]
    else:
        data = [[int(sbox[i*16 + j]) for j in range(16)] for i in range(16)]
    
    df = pd.DataFrame(data, 
                     columns=[f"{i:X}" for i in range(16)],
                     index=[f"{i:X}0" for i in range(16)])
    
    st.dataframe(df, use_container_width=True)
    
    # Download options
    st.markdown("### ⬇️ Download S-Box Data")
    col1, col2 = st.columns(2)
    
    with col1:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer)
        st.download_button(
            label="📄 Download as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"{sbox_name.replace('📝 ', '').replace('🖼️ ', '')}_sbox_{display_format.lower()}.csv",
            mime="text/csv",
            key="download_csv",
            width='stretch'  # or width='stretch' for latest Streamlit

        )
    
    with col2:
        json_data = {
            'name': sbox_name,
            'format': display_format,
            'sbox_flat': sbox.tolist(),
            'sbox_2d': [[int(sbox[i*16 + j]) for j in range(16)] for i in range(16)]
        }
        st.download_button(
            label="📋 Download as JSON",
            data=json.dumps(json_data, indent=4),
            file_name=f"{sbox_name.replace('📝 ', '').replace('🖼️ ', '')}_sbox.json",
            mime="application/json",
            key="download_json",
            use_container_width=True
        )