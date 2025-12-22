from flask import Flask, render_template, request, send_file, jsonify
import io
import csv
import numpy as np
import random
import hashlib
import json
import os
from pathlib import Path
from sbox_utils import generate_sboxes, balance, bijective, nonlinearity, sac, lap, dap, bic_sac_fast, bic_nl_fast, show_test_process, compare_with_ideal
from crypto import AESCustom

app = Flask(__name__)
app.secret_key = 'replace-with-your-secret-key'

# Path untuk file hasil metrics
METRICS_FILE = Path(__file__).parent / 'sbox_results.json'

# Cache global untuk menyimpan hasil metrics
METRICS_CACHE = {}
METRICS_BY_NAME = {}  # Cache berdasarkan nama S-box

def load_metrics_from_file():
    """Load pre-calculated metrics dari file JSON"""
    global METRICS_CACHE, METRICS_BY_NAME
    
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE, 'r') as f:
                data = json.load(f)
                
                # Load data dari format file Anda
                METRICS_BY_NAME = data
                
                # Build hash-based cache untuk compatibility
                sboxes = generate_sboxes(include_random=False)
                for name, sbox in sboxes.items():
                    if name in METRICS_BY_NAME:
                        sbox_hash = get_sbox_hash(sbox)
                        # Remove "S-box" key dan convert ke format internal
                        metrics = {k: v for k, v in METRICS_BY_NAME[name].items() if k != "S-box"}
                        METRICS_CACHE[sbox_hash] = metrics
                
                print(f"✅ Loaded {len(METRICS_BY_NAME)} pre-calculated metrics from {METRICS_FILE.name}")
                return True
        except Exception as e:
            print(f"⚠️  Error loading metrics file: {e}")
            METRICS_CACHE = {}
            METRICS_BY_NAME = {}
            return False
    else:
        print(f"⚠️  Metrics file not found: {METRICS_FILE}")
        METRICS_CACHE = {}
        METRICS_BY_NAME = {}
        return False

def save_metrics_to_file():
    """Save calculated metrics ke file JSON"""
    try:
        # Convert METRICS_BY_NAME back to file format
        data = {}
        for name, metrics in METRICS_BY_NAME.items():
            data[name] = {
                "S-box": name,
                **metrics
            }
        
        with open(METRICS_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"💾 Saved {len(data)} metrics to {METRICS_FILE.name}")
        return True
    except Exception as e:
        print(f"❌ Error saving metrics file: {e}")
        return False

def get_sbox_hash(sbox):
    """Generate unique hash untuk S-box"""
    return hashlib.md5(sbox.tobytes()).hexdigest()

def sbox_metrics(sbox, sbox_name=None, force_recalculate=False):
    """Hitung metrics S-box dengan caching"""
    # Try to get from name cache first
    if not force_recalculate and sbox_name and sbox_name in METRICS_BY_NAME:
        return METRICS_BY_NAME[sbox_name], True
    
    # Try hash-based cache
    sbox_hash = get_sbox_hash(sbox)
    if not force_recalculate and sbox_hash in METRICS_CACHE:
        return METRICS_CACHE[sbox_hash], True
    
    # Calculate metrics
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
    
    # Convert numpy types to native Python types for JSON serialization
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            metrics_serializable[key] = float(value)
        elif isinstance(value, np.bool_):
            metrics_serializable[key] = bool(value)
        else:
            metrics_serializable[key] = value
    
    # Cache it
    METRICS_CACHE[sbox_hash] = metrics_serializable
    if sbox_name:
        METRICS_BY_NAME[sbox_name] = metrics_serializable
    
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
    
    # Ensure input is string
    if isinstance(user_input, bytes):
        user_input = user_input.decode('utf-8', errors='ignore')
    
    user_input = str(user_input).strip()
    
    # Remove spaces for hex detection
    user_input_no_space = user_input.replace(" ", "")
    
    try:
        # Try to parse as hex (only if it looks like hex)
        if all(c in '0123456789abcdefABCDEF' for c in user_input_no_space):
            b = bytes.fromhex(user_input_no_space)
        else:
            # Otherwise encode as UTF-8
            b = user_input.encode('utf-8')
    except ValueError:
        # If hex parsing fails, encode as UTF-8
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
            # Try parsing with base auto-detect
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
    
    # Validate uniqueness
    if len(set(sbox_list)) != 256:
        raise ValueError("Custom S-box must contain 256 unique values (bijective requirement)")
    
    return np.array(sbox_list, dtype=np.uint8)

def calculate_std_metrics(sboxes_dict=None):
    """Calculate standard deviation and other statistics for each metric across all S-boxes"""
    metrics_by_type = {}
    
    # If no sboxes provided, use the ones from file
    if sboxes_dict is None:
        data_source = METRICS_BY_NAME
    else:
        # Calculate from provided sboxes
        data_source = {}
        for name, sbox in sboxes_dict.items():
            metrics, _ = sbox_metrics(sbox, sbox_name=name, force_recalculate=False)
            data_source[name] = metrics
    
    # Collect values for each metric
    for name, metrics in data_source.items():
        for metric_name, value in metrics.items():
            if metric_name not in ["Balance", "Bijective", "S-box"]:  # Skip boolean and metadata
                if metric_name not in metrics_by_type:
                    metrics_by_type[metric_name] = []
                metrics_by_type[metric_name].append(float(value))
    
    # Calculate statistics for each metric
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

@app.route('/', methods=['GET', 'POST'])
def index():
    sboxes = generate_sboxes(include_random=True)
    result = ''
    result_decimal = ''
    result_text = ''
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

        # Hitung metrics S-box aktif
        if active_sbox is not None and action not in ["Encrypt", "Decrypt", "Download S-Box CSV"]:
            try:
                metrics, from_cache = sbox_metrics(active_sbox, sbox_name=sbox_choice, force_recalculate=False)
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
                aes.sbox = active_sbox
                
                # Generate inverse S-box
                aes.inv_sbox = np.zeros(256, dtype=np.uint8)
                for i in range(256):
                    aes.inv_sbox[aes.sbox[i]] = i
                
                if action == "Encrypt":
                    ct_bytes = aes.encrypt(pt_bytes)
                    result = ct_bytes.hex()
                    result_decimal = ' '.join(str(b) for b in ct_bytes)
                    try:
                        result_text = ct_bytes.decode('utf-8', errors='replace')
                    except:
                        result_text = '(non-printable)'
                else:  # Decrypt
                    pt_bytes2 = aes.decrypt(pt_bytes)
                    result = pt_bytes2.hex()
                    result_decimal = ' '.join(str(b) for b in pt_bytes2)
                    try:
                        result_text = pt_bytes2.decode('utf-8', errors='replace')
                    except:
                        result_text = '(non-printable)'
                
                plaintext = plaintext_input
                key = key_input
                
            except Exception as e:
                result = f"❌ Error during {action.lower()}: {str(e)}"
                import traceback
                print(f"Encryption/Decryption Error: {traceback.format_exc()}")

    return render_template('index.html',
                           result=result,
                           result_decimal=result_decimal,
                           result_text=result_text,
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
    """API endpoint untuk mendapatkan comparison metrics semua S-boxes dengan STD"""
    try:
        force_recalculate = request.args.get('recalculate', 'false').lower() == 'true'
        
        sboxes = generate_sboxes(include_random=False)
        comparison_data = {}
        
        for name, sbox in sboxes.items():
            metrics, _ = sbox_metrics(sbox, sbox_name=name, force_recalculate=force_recalculate)
            
            # Metrics already in native Python types from sbox_metrics
            metrics_json = metrics
            
            # Calculate overall score
            scores = []
            for key, value in metrics_json.items():
                if key not in ["Balance", "Bijective"]:
                    scores.append(get_metric_score(key, value))
            
            overall_score = sum(scores) / len(scores) if scores else 0
            
            comparison_data[name] = {
                'metrics': metrics_json,
                'overall_score': round(overall_score, 2)
            }
        
        # Calculate STD and statistics for each metric
        std_metrics = calculate_std_metrics(sboxes if force_recalculate else None)
        
        # Save to file if recalculated
        if force_recalculate:
            save_metrics_to_file()
        
        return jsonify({
            'data': comparison_data,
            'statistics': std_metrics,  # Changed from 'std' to 'statistics' for clarity
            'from_cache': not force_recalculate
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recalculate-all', methods=['POST'])
def recalculate_all_metrics():
    """API endpoint untuk recalculate semua metrics dan save ke file"""
    try:
        sboxes = generate_sboxes(include_random=False)
        recalculated_count = 0
        
        print("\n🔄 Recalculating all S-box metrics...")
        
        for name, sbox in sboxes.items():
            print(f"   Computing {name}...", end=" ", flush=True)
            try:
                metrics, _ = sbox_metrics(sbox, sbox_name=name, force_recalculate=True)
                recalculated_count += 1
                print("✓")
            except Exception as e:
                print(f"✗ Error: {e}")
        
        # Calculate statistics
        std_metrics = calculate_std_metrics(sboxes)
        
        # Save to file
        if save_metrics_to_file():
            return jsonify({
                'success': True,
                'message': f'Successfully recalculated {recalculated_count} S-boxes',
                'count': recalculated_count,
                'statistics': std_metrics
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Recalculated but failed to save to file',
                'count': recalculated_count
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """API endpoint untuk mendapatkan statistik (mean, std, min, max, median) dari semua metrics"""
    try:
        stats = calculate_std_metrics()
        return jsonify({
            'success': True,
            'statistics': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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

@app.route('/api/test-process/<sbox_name>', methods=['GET'])
def get_test_process(sbox_name):
    """API endpoint untuk mendapatkan detail proses testing S-box"""
    try:
        sboxes = generate_sboxes(include_random=False)
        
        if sbox_name not in sboxes:
            return jsonify({'error': 'S-box not found'}), 404
        
        sbox = sboxes[sbox_name]
        
        # Run detailed test process
        results = show_test_process(sbox_name, sbox, verbose=False)
        
        # Convert numpy types to native Python types
        results_native = {}
        for key, value in results.items():
            if isinstance(value, (np.integer, np.floating)):
                results_native[key] = float(value)
            elif isinstance(value, np.bool_):
                results_native[key] = bool(value)
            else:
                results_native[key] = value
        
        # Calculate quality scores
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
        
        quality_scores = {}
        scores = []
        
        for metric, (ideal, comparison_type) in ideals.items():
            value = results_native[metric]
            
            if comparison_type == 'boolean':
                score = 100 if value == ideal else 0
                status = "Pass" if value == ideal else "Fail"
            elif comparison_type == 'higher':
                score = min(100, (value / ideal) * 100) if ideal > 0 else 0
                status = "Excellent" if value >= ideal else "Good" if value >= ideal * 0.9 else "Needs Improvement"
            elif comparison_type == 'closer':
                deviation = abs(value - ideal)
                score = max(0, 100 - (deviation * 1000))
                status = "Excellent" if deviation < 0.01 else "Good" if deviation < 0.02 else "Needs Improvement"
            elif comparison_type == 'lower_equal':
                if value <= ideal:
                    score = 100
                    status = "Excellent"
                else:
                    score = max(0, 100 - ((value - ideal) / ideal * 100))
                    status = "Acceptable" if value <= ideal * 1.2 else "Needs Improvement"
            
            quality_scores[metric] = {
                'value': value,
                'ideal': ideal,
                'score': score,
                'status': status
            }
            scores.append(score)
        
        overall_score = sum(scores) / len(scores) if scores else 0
        
        if overall_score >= 90:
            grade = "A+"
        elif overall_score >= 80:
            grade = "A"
        elif overall_score >= 70:
            grade = "B"
        elif overall_score >= 60:
            grade = "C"
        else:
            grade = "D"
        
        return jsonify({
            'sbox_name': sbox_name,
            'results': results_native,
            'quality_scores': quality_scores,
            'overall_score': round(overall_score, 2),
            'grade': grade
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-all-sboxes', methods=['GET'])
def test_all_sboxes():
    """API endpoint untuk test semua S-boxes dan return detailed process"""
    try:
        sboxes = generate_sboxes(include_random=False)
        all_results = {}
        
        for name, sbox in sboxes.items():
            # Get test results
            results = show_test_process(name, sbox, verbose=False)
            
            # Convert numpy types
            results_native = {}
            for key, value in results.items():
                if isinstance(value, (np.integer, np.floating)):
                    results_native[key] = float(value)
                elif isinstance(value, np.bool_):
                    results_native[key] = bool(value)
                else:
                    results_native[key] = value
            
            all_results[name] = results_native
        
        return jsonify({
            'success': True,
            'results': all_results,
            'message': 'All S-boxes tested successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    import os
    
    print("🚀 Initializing AES Encryption Suite...")
    print("=" * 50)
    
    # Load pre-calculated metrics from file
    if load_metrics_from_file():
        print("⚡ Ready! Using cached metrics for instant access.")
        
        # Show statistics
        try:
            stats = calculate_std_metrics()
            print(f"\n📊 Loaded Statistics Summary:")
            for metric, values in stats.items():
                print(f"   {metric:10s} → STD: {values['std']:.6f}, Mean: {values['mean']:.6f}")
        except:
            pass
    else:
        print("📊 No cached metrics found. Calculating on first use...")
        print("💡 Tip: Use 'Recalculate All Metrics' to generate cache file.")
    
    print("=" * 50)
    
    # Check if running in production
    if os.environ.get('FLASK_ENV') == 'production' or os.environ.get('PORT'):
        # Production mode (Render, Railway, etc)
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port)
    else:
        # Development mode
        app.run(debug=True, use_reloader=False)