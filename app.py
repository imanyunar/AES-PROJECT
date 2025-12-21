from flask import Flask, render_template, request, send_file, jsonify
import io
import csv
import numpy as np
import random
import hashlib
from sbox_utils import generate_sboxes, balance, bijective, nonlinearity, sac, lap, dap, bic_sac_fast, bic_nl_fast
from crypto import AESCustom

app = Flask(__name__)
app.secret_key = 'replace-with-your-secret-key'  # Ganti dengan key aman

# Cache global untuk menyimpan hasil metrics
# Format: {sbox_hash: {metrics_dict}}
METRICS_CACHE = {}

def get_sbox_hash(sbox):
    """Generate unique hash untuk S-box"""
    return hashlib.md5(sbox.tobytes()).hexdigest()

def sbox_metrics(sbox, force_recalculate=False):
    """Hitung metrics S-box dengan caching"""
    sbox_hash = get_sbox_hash(sbox)
    if not force_recalculate and sbox_hash in METRICS_CACHE:
        return METRICS_CACHE[sbox_hash], True
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
    METRICS_CACHE[sbox_hash] = metrics
    return metrics, False

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
        return 50

def parse_user_bytes(user_input, length=16):
    """Convert user input to bytes safely, truncate/pad ke length"""
    if not user_input:
        return bytes([0]*length)
    
    # Ensure input is string
    if isinstance(user_input, bytes):
        user_input = user_input.decode('utf-8', errors='ignore')
    
    user_input = str(user_input).strip().replace(" ", "")
    
    try:
        # Try to parse as hex
        b = bytes.fromhex(user_input)
    except ValueError:
        # If not hex, encode as UTF-8
        b = user_input.encode('utf-8')
    
    # Adjust length
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
    
    # Split by comma
    parts = custom_sbox_str.split(',')
    
    for i, part in enumerate(parts):
        part = part.strip()
        
        if not part:
            continue
        
        try:
            # Try parsing with base auto-detect (0x for hex, 0o for octal, etc.)
            val = int(part, 0)
        except ValueError:
            try:
                # Try parsing as base 10
                val = int(part)
            except ValueError:
                raise ValueError(f"Cannot parse value at position {i}: '{part}'")
        
        # Validate range
        if not (0 <= val <= 255):
            raise ValueError(f"Value at position {i} is out of range (0-255): {val}")
        
        sbox_list.append(val)
    
    # Validate length
    if len(sbox_list) != 256:
        raise ValueError(f"Custom S-box must have exactly 256 elements, got {len(sbox_list)}")
    
    # Validate uniqueness (bijective requirement)
    if len(set(sbox_list)) != 256:
        raise ValueError("Custom S-box must contain 256 unique values (bijective requirement)")
    
    return np.array(sbox_list, dtype=np.uint8)

@app.route('/', methods=['GET', 'POST'])
def index():
    sboxes = generate_sboxes(include_random=True)
    result = ''
    metrics = {}
    plaintext = ''
    key = ''
    sbox_choice = 'K44'
    custom_sbox = ''
    active_sbox = None
    from_cache = False
    show_recalculate_button = False

    if request.method == 'POST':
        action = request.form.get('action')
        sbox_choice = request.form.get('sbox_choice')

        # Tentukan S-box aktif
        if sbox_choice in sboxes:
            active_sbox = sboxes[sbox_choice]
        elif sbox_choice == "RANDOM":
            active_sbox = np.array(random.sample(range(256), 256), dtype=np.uint8)
        elif sbox_choice == "CUSTOM":
            custom_sbox_str = request.form.get('custom_sbox', '')
            custom_sbox = custom_sbox_str
            try:
                active_sbox = parse_custom_sbox(custom_sbox_str)
            except Exception as e:
                result = f"❌ Error parsing custom S-box: {str(e)}"
                active_sbox = None
        else:
            result = "❌ S-box tidak dikenali"
            active_sbox = None

        # Download S-box CSV
        if action == "Download S-Box CSV" and active_sbox is not None:
            try:
                proxy = io.StringIO()
                writer = csv.writer(proxy)
                for val in active_sbox:
                    writer.writerow([val])
                mem = io.BytesIO()
                mem.write(proxy.getvalue().encode('utf-8'))
                mem.seek(0)
                proxy.close()
                filename = f"{sbox_choice}_sbox.csv"
                return send_file(mem, as_attachment=True, download_name=filename, mimetype='text/csv')
            except Exception as e:
                result = f"❌ Error downloading S-box: {str(e)}"

        # Recalculate All Metrics
        if action == "Recalculate All Metrics" and active_sbox is not None:
            try:
                metrics, from_cache = sbox_metrics(active_sbox, force_recalculate=True)
                result = "✅ Metrics berhasil dihitung ulang!"
                show_recalculate_button = True
            except Exception as e:
                result = f"❌ Error calculating metrics: {str(e)}"

        # Hitung metrics S-box aktif (tanpa recalc)
        elif active_sbox is not None and action not in ["Encrypt", "Decrypt"]:
            try:
                metrics, from_cache = sbox_metrics(active_sbox, force_recalculate=False)
                show_recalculate_button = True
            except Exception as e:
                result = f"❌ Error calculating metrics: {str(e)}"

        # Encrypt / Decrypt
        if action in ["Encrypt", "Decrypt"] and active_sbox is not None:
            plaintext_input = request.form.get('plaintext', '')
            key_input = request.form.get('key', '')
            
            try:
                pt_bytes = parse_user_bytes(plaintext_input, length=16)
                key_bytes = parse_user_bytes(key_input, length=16)
                
                aes = AESCustom(sbox_name=sbox_choice, key_bytes=key_bytes)
                aes.sbox = active_sbox  # Override jika random/custom
                
                # Generate inverse S-box
                aes.inv_sbox = np.zeros(256, dtype=np.uint8)
                for i in range(256):
                    aes.inv_sbox[aes.sbox[i]] = i
                
                if action == "Encrypt":
                    ct_bytes = aes.encrypt(pt_bytes)
                    result = ct_bytes.hex()
                else:  # Decrypt
                    pt_bytes2 = aes.decrypt(pt_bytes)
                    result = pt_bytes2.hex()
                
                plaintext = pt_bytes.hex()
                key = key_bytes.hex()
                
            except Exception as e:
                result = f"❌ Error during {action.lower()}: {str(e)}"
                import traceback
                print(f"Encryption/Decryption Error: {traceback.format_exc()}")

    return render_template('index.html',
                           result=result,
                           plaintext=plaintext,
                           key=key,
                           sbox_choice=sbox_choice,
                           custom_sbox=custom_sbox,
                           metrics=metrics,
                           from_cache=from_cache,
                           show_recalculate_button=show_recalculate_button,
                           calculate_progress_width=calculate_progress_width,
                           active_sbox=active_sbox
                           )

@app.route('/api/compare-sboxes', methods=['GET'])
def compare_sboxes():
    """API endpoint untuk mendapatkan comparison metrics semua S-boxes"""
    try:
        sboxes = generate_sboxes(include_random=False)  # Exclude random untuk consistency
        comparison_data = {}
        
        for name, sbox in sboxes.items():
            metrics, _ = sbox_metrics(sbox)
            
            # Convert numpy types to Python native types for JSON serialization
            metrics_json = {}
            for key, value in metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    metrics_json[key] = float(value)
                elif isinstance(value, np.bool_):
                    metrics_json[key] = bool(value)
                else:
                    metrics_json[key] = value
            
            # Calculate overall score
            scores = []
            for key, value in metrics_json.items():
                if key not in ["Balance", "Bijective"]:  # Skip boolean metrics
                    scores.append(get_metric_score(key, value))
            
            overall_score = sum(scores) / len(scores) if scores else 0
            
            comparison_data[name] = {
                'metrics': metrics_json,
                'overall_score': round(overall_score, 2)
            }
        
        return jsonify(comparison_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sbox-data/<sbox_name>', methods=['GET'])
def get_sbox_data(sbox_name):
    """API endpoint untuk mendapatkan data S-box dalam hex dan decimal"""
    try:
        sboxes = generate_sboxes(include_random=False)
        
        if sbox_name not in sboxes:
            return jsonify({'error': 'S-box not found'}), 404
        
        sbox = sboxes[sbox_name]
        
        # Format data in 16x16 grid
        sbox_data = {
            'name': sbox_name,
            'hex': [[f"{sbox[i*16 + j]:02X}" for j in range(16)] for i in range(16)],
            'dec': [[int(sbox[i*16 + j]) for j in range(16)] for i in range(16)]
        }
        
        return jsonify(sbox_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Pre-calculate metrics untuk S-box default saat startup
    print("🚀 Initializing AES Encryption Suite...")
    print("📊 Pre-calculating metrics for default S-boxes...")
    sboxes = generate_sboxes(include_random=True)
    for name, sbox in sboxes.items():
        print(f"   Computing {name}...", end=" ")
        try:
            metrics, _ = sbox_metrics(sbox)
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
    print("✅ Ready! Metrics cached for instant access.")
    print("=" * 50)
    # Disable reloader to avoid double initialization
    app.run(debug=True, use_reloader=False)