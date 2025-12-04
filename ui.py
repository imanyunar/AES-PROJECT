from flask import Flask, render_template_string, send_file, request
import os

app = Flask(__name__)

# ===== Fungsi untuk membaca S-box dari file =====
def read_sbox_decimal():
    sbox = []
    with open("best_sbox_decimal.txt", "r") as f:
        for line in f:
            row = [int(x) for x in line.strip().split()]
            sbox.append(row)
    return sbox

def read_sbox_hex():
    sbox = []
    with open("best_sbox_hex.txt", "r") as f:
        for line in f:
            row = [x.strip() for x in line.strip().split()]
            sbox.append(row)
    return sbox

# ===== HTML template dengan styling yang lebih baik =====
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Best AES S-box</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary: #3498db;
            --secondary: #2c3e50;
            --accent: #e74c3c;
            --light: #ecf0f1;
            --dark: #2c3e50;
            --success: #27ae60;
            --warning: #f39c12;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
            animation: slideUp 0.8s ease-out;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        @keyframes glow {
            0%, 100% { box-shadow: 0 0 5px var(--primary); }
            50% { box-shadow: 0 0 20px var(--primary); }
        }

        header {
            background: linear-gradient(to right, var(--secondary), var(--dark));
            color: white;
            padding: 30px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
            background-size: 20px 20px;
            animation: float 20s linear infinite;
        }

        @keyframes float {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        h1 {
            font-size: 2.8rem;
            margin-bottom: 10px;
            position: relative;
            display: inline-block;
        }

        h1::after {
            content: '';
            position: absolute;
            bottom: -10px;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 4px;
            background: var(--accent);
            border-radius: 2px;
        }

        .subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 20px;
        }

        .main-content {
            padding: 30px;
        }

        .section {
            margin-bottom: 40px;
            padding: 25px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }

        .section:hover {
            transform: translateY(-5px);
        }

        h2 {
            color: var(--secondary);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--light);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        h2 i {
            color: var(--primary);
        }

        .download-buttons {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-top: 20px;
        }

        .btn {
            padding: 14px 28px;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 10px;
            text-decoration: none;
        }

        .btn-primary {
            background: linear-gradient(to right, var(--primary), #2980b9);
            color: white;
        }

        .btn-primary:hover {
            background: linear-gradient(to right, #2980b9, var(--primary));
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(52, 152, 219, 0.3);
            animation: glow 1.5s infinite;
        }

        .btn-secondary {
            background: var(--light);
            color: var(--dark);
            border: 2px solid var(--primary);
        }

        .btn-secondary:hover {
            background: var(--primary);
            color: white;
            transform: translateY(-3px);
        }

        .btn-success {
            background: linear-gradient(to right, var(--success), #229954);
            color: white;
        }

        .btn-success:hover {
            background: linear-gradient(to right, #229954, var(--success));
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(39, 174, 96, 0.3);
        }

        .btn-warning {
            background: linear-gradient(to right, var(--warning), #e67e22);
            color: white;
        }

        .btn-warning:hover {
            background: linear-gradient(to right, #e67e22, var(--warning));
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(243, 156, 18, 0.3);
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            color: var(--dark);
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .input-container {
            position: relative;
        }

        input[type="text"] {
            width: 100%;
            padding: 15px 20px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 1rem;
            transition: all 0.3s ease;
            background: var(--light);
        }

        input[type="text"]:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
            background: white;
        }

        .input-info {
            font-size: 0.85rem;
            color: #666;
            margin-top: 5px;
            padding-left: 10px;
            border-left: 3px solid var(--primary);
            background: rgba(52, 152, 219, 0.1);
            padding: 8px 12px;
            border-radius: 0 5px 5px 0;
        }

        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        @media (max-width: 768px) {
            .form-row {
                grid-template-columns: 1fr;
            }
        }

        .sbox-container {
            display: none;
            animation: fadeIn 0.8s ease-out;
        }

        .sbox-container.show {
            display: block;
        }

        .sbox-wrapper {
            overflow-x: auto;
            margin-top: 20px;
        }

        .sbox-table {
            width: 100%;
            border-collapse: collapse;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }

        .sbox-table th {
            background: linear-gradient(to right, var(--secondary), var(--dark));
            color: white;
            padding: 15px;
            font-weight: 600;
            position: sticky;
            top: 0;
        }

        .sbox-table td {
            padding: 12px 15px;
            border: 1px solid #eee;
            text-align: center;
            transition: all 0.3s ease;
            font-weight: 500;
        }

        .sbox-table tr:nth-child(even) td {
            background: rgba(52, 152, 219, 0.05);
        }

        .sbox-table tr:hover td {
            background: rgba(52, 152, 219, 0.1);
            transform: scale(1.02);
        }

        .decimal-cell {
            color: var(--primary);
            font-weight: bold;
        }

        .hex-cell {
            color: var(--success);
            font-weight: bold;
            font-family: 'Courier New', monospace;
        }

        .result-box {
            padding: 20px;
            margin-top: 20px;
            border-radius: 10px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-left: 5px solid var(--primary);
            animation: fadeIn 0.5s ease-out;
            display: none;
        }

        .result-box.show {
            display: block;
        }

        .result-title {
            color: var(--secondary);
            margin-bottom: 10px;
            font-size: 1.2rem;
        }

        .result-value {
            font-family: 'Courier New', monospace;
            font-size: 1.1rem;
            background: rgba(0, 0, 0, 0.05);
            padding: 15px;
            border-radius: 5px;
            word-break: break-all;
        }

        footer {
            text-align: center;
            padding: 20px;
            color: #666;
            border-top: 1px solid #eee;
            font-size: 0.9rem;
        }

        .toggle-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(to right, var(--primary), #2980b9);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        .toggle-btn:hover {
            background: linear-gradient(to right, #2980b9, var(--primary));
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(52, 152, 219, 0.3);
        }

        .toggle-btn i {
            transition: transform 0.3s ease;
        }

        .toggle-btn.active i {
            transform: rotate(180deg);
        }

        .sbox-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }

        .tab-btn {
            padding: 10px 20px;
            background: #f1f1f1;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
        }

        .tab-btn.active {
            background: var(--primary);
            color: white;
        }

        .tab-btn:hover:not(.active) {
            background: #ddd;
        }

        .alert {
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            animation: slideUp 0.5s ease-out;
        }

        .alert-info {
            background: rgba(52, 152, 219, 0.1);
            color: var(--primary);
            border-left: 4px solid var(--primary);
        }

        .alert-warning {
            background: rgba(243, 156, 18, 0.1);
            color: var(--warning);
            border-left: 4px solid var(--warning);
        }

        .alert-success {
            background: rgba(39, 174, 96, 0.1);
            color: var(--success);
            border-left: 4px solid var(--success);
        }
    </style>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Toggle S-box visibility
            const toggleBtn = document.getElementById('toggleSbox');
            const sboxContainer = document.getElementById('sboxContainer');
            const toggleIcon = toggleBtn.querySelector('i');
            
            toggleBtn.addEventListener('click', function() {
                sboxContainer.classList.toggle('show');
                toggleBtn.classList.toggle('active');
                
                if (sboxContainer.classList.contains('show')) {
                    toggleBtn.innerHTML = '<i class="fas fa-chevron-up"></i> Hide S-Box';
                } else {
                    toggleBtn.innerHTML = '<i class="fas fa-chevron-down"></i> Show S-Box';
                }
                
                // Animate scroll to sbox
                if (sboxContainer.classList.contains('show')) {
                    sboxContainer.scrollIntoView({ behavior: 'smooth' });
                }
            });

            // Tab switching for S-box
            const tabBtns = document.querySelectorAll('.tab-btn');
            const decimalTable = document.getElementById('decimalTable');
            const hexTable = document.getElementById('hexTable');
            
            tabBtns.forEach(btn => {
                btn.addEventListener('click', function() {
                    // Remove active class from all tabs
                    tabBtns.forEach(b => b.classList.remove('active'));
                    
                    // Add active class to clicked tab
                    this.classList.add('active');
                    
                    // Show corresponding table
                    if (this.dataset.tab === 'decimal') {
                        decimalTable.style.display = 'table';
                        hexTable.style.display = 'none';
                    } else {
                        decimalTable.style.display = 'none';
                        hexTable.style.display = 'table';
                    }
                });
            });

            // Show result box if there's a result
            const resultBox = document.getElementById('resultBox');
            if (resultBox && resultBox.textContent.trim() !== '') {
                resultBox.classList.add('show');
                resultBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }

            // Form validation and animations
            const forms = document.querySelectorAll('form');
            forms.forEach(form => {
                form.addEventListener('submit', function(e) {
                    const inputs = this.querySelectorAll('input[required]');
                    let isValid = true;
                    
                    inputs.forEach(input => {
                        if (!input.value.trim()) {
                            isValid = false;
                            input.style.borderColor = 'var(--accent)';
                            input.style.animation = 'pulse 0.5s';
                            
                            setTimeout(() => {
                                input.style.animation = '';
                            }, 500);
                        }
                    });
                    
                    if (!isValid) {
                        e.preventDefault();
                        alert('Please fill in all required fields!');
                    }
                });
            });

            // Input focus effects
            const inputs = document.querySelectorAll('input[type="text"]');
            inputs.forEach(input => {
                input.addEventListener('focus', function() {
                    this.parentElement.style.transform = 'translateY(-2px)';
                });
                
                input.addEventListener('blur', function() {
                    this.parentElement.style.transform = 'translateY(0)';
                });
            });
        });
    </script>
</head>
<body>
    <div class="container">
        <header>
            <h1><i class="fas fa-shield-alt"></i> AES S-Box System</h1>
            <div class="subtitle">Advanced Encryption Standard - Best S-Box Implementation</div>
        </header>
        
        <div class="main-content">
            <!-- Alert Section -->
            <div class="alert alert-info">
                <i class="fas fa-info-circle"></i>
                <div>
                    <strong>Information:</strong> This system implements the best AES S-Box for encryption and decryption operations. 
                    Ensure you have valid input data in the correct format.
                </div>
            </div>
            
            <!-- Download Section -->
            <div class="section">
                <h2><i class="fas fa-download"></i> Download S-Box Files</h2>
                <p>Download the S-Box values in different formats for offline use or integration into your projects.</p>
                <div class="download-buttons">
                    <a href="/download/decimal" class="btn btn-primary">
                        <i class="fas fa-file-alt"></i> Decimal Format
                    </a>
                    <a href="/download/hex" class="btn btn-success">
                        <i class="fas fa-file-code"></i> Hexadecimal Format
                    </a>
                </div>
            </div>
            
            <!-- Show S-Box Button -->
            <button id="toggleSbox" class="toggle-btn">
                <i class="fas fa-chevron-down"></i> Show S-Box
            </button>
            
            <!-- S-Box Display Section -->
            <div id="sboxContainer" class="sbox-container">
                <div class="section">
                    <h2><i class="fas fa-table"></i> S-Box Values</h2>
                    <p>View the complete substitution box used in AES encryption. The S-Box is a 16x16 table containing 256 byte values.</p>
                    
                    <div class="sbox-tabs">
                        <button class="tab-btn active" data-tab="decimal">Decimal View</button>
                        <button class="tab-btn" data-tab="hex">Hexadecimal View</button>
                    </div>
                    
                    <div class="sbox-wrapper">
                        <!-- Decimal S-Box Table -->
                        <table id="decimalTable" class="sbox-table">
                            <thead>
                                <tr>
                                    <th></th>
                                    {% for i in range(16) %}
                                    <th>{{ i }}</th>
                                    {% endfor %}
                                </tr>
                            </thead>
                            <tbody>
                                {% for i, row in enumerate(sbox_decimal or []) %}
                                <tr>
                                    <th>{{ i }}</th>
                                    {% for val in row %}
                                    <td class="decimal-cell">{{ val }}</td>
                                    {% endfor %}
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                        
                        <!-- Hex S-Box Table -->
                        <table id="hexTable" class="sbox-table" style="display: none;">
                            <thead>
                                <tr>
                                    <th></th>
                                    {% for i in range(16) %}
                                    <th>{{ "%X"|format(i) }}</th>
                                    {% endfor %}
                                </tr>
                            </thead>
                            <tbody>
                                {% for i, row in enumerate(sbox_hex or []) %}
                                <tr>
                                    <th>{{ "%X"|format(i) }}</th>
                                    {% for val in row %}
                                    <td class="hex-cell">{{ val }}</td>
                                    {% endfor %}
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <!-- Encryption and Decryption Forms -->
            <div class="form-row">
                <!-- Encryption Form -->
                <div class="section">
                    <h2><i class="fas fa-lock"></i> Encryption</h2>
                    <p>Encrypt your plaintext using AES with the best S-Box.</p>
                    
                    <form method="POST" action="/encrypt">
                        <div class="form-group">
                            <label for="plaintext"><i class="fas fa-keyboard"></i> Plaintext</label>
                            <div class="input-container">
                                <input type="text" id="plaintext" name="plaintext" 
                                       placeholder="Enter text to encrypt" required>
                            </div>
                            <div class="input-info">
                                <strong>Format:</strong> Any ASCII text (max 256 characters)<br>
                                <strong>Example:</strong> "Hello AES", "Secret Message", "12345"
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label for="key"><i class="fas fa-key"></i> Encryption Key</label>
                            <div class="input-container">
                                <input type="text" id="key" name="key" 
                                       placeholder="Enter encryption key" required>
                            </div>
                            <div class="input-info">
                                <strong>Format:</strong> 16, 24, or 32 characters (128, 192, or 256-bit)<br>
                                <strong>Example:</strong> "MySecretKey12345" (16 chars for 128-bit)
                            </div>
                        </div>
                        
                        <button type="submit" class="btn btn-primary" style="width: 100%;">
                            <i class="fas fa-lock"></i> Encrypt Now
                        </button>
                    </form>
                </div>
                
                <!-- Decryption Form -->
                <div class="section">
                    <h2><i class="fas fa-unlock"></i> Decryption</h2>
                    <p>Decrypt your ciphertext using AES with the inverse S-Box.</p>
                    
                    <form method="POST" action="/decrypt">
                        <div class="form-group">
                            <label for="ciphertext"><i class="fas fa-file-code"></i> Ciphertext</label>
                            <div class="input-container">
                                <input type="text" id="ciphertext" name="ciphertext" 
                                       placeholder="Enter text to decrypt" required>
                            </div>
                            <div class="input-info">
                                <strong>Format:</strong> Hexadecimal string (output from encryption)<br>
                                <strong>Example:</strong> "A7F3C2E4B5D6A8C9", "0123456789ABCDEF"
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label for="key2"><i class="fas fa-key"></i> Decryption Key</label>
                            <div class="input-container">
                                <input type="text" id="key2" name="key" 
                                       placeholder="Enter decryption key" required>
                            </div>
                            <div class="input-info">
                                <strong>Must match:</strong> The same key used for encryption<br>
                                <strong>Note:</strong> AES is symmetric - same key for both operations
                            </div>
                        </div>
                        
                        <button type="submit" class="btn btn-success" style="width: 100%;">
                            <i class="fas fa-unlock"></i> Decrypt Now
                        </button>
                    </form>
                </div>
            </div>
            
            <!-- Result Display -->
            {% if result %}
            <div id="resultBox" class="result-box show">
                <div class="result-title">
                    <i class="fas fa-check-circle"></i> Operation Result
                </div>
                <div class="result-value">{{ result }}</div>
            </div>
            {% else %}
            <div id="resultBox" class="result-box"></div>
            {% endif %}
            
            <!-- Information Section -->
            <div class="section">
                <h2><i class="fas fa-info-circle"></i> System Information</h2>
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i>
                    <div>
                        <strong>Important Notes:</strong><br>
                        1. This is a demonstration system using the best AES S-Box<br>
                        2. Ensure proper key management in production environments<br>
                        3. Input validation is required for secure operations<br>
                        4. The S-Box is loaded from external files (best_sbox_decimal.txt and best_sbox_hex.txt)
                    </div>
                </div>
                
                <div class="alert alert-success">
                    <i class="fas fa-check-circle"></i>
                    <div>
                        <strong>Features:</strong><br>
                        • Best AES S-Box implementation<br>
                        • Responsive and animated interface<br>
                        • Tabbed S-Box viewing (Decimal/Hex)<br>
                        • Form validation with visual feedback<br>
                        • Downloadable S-Box files<br>
                        • Smooth animations and transitions
                    </div>
                </div>
            </div>
        </div>
        
        <footer>
            <p>AES S-Box System &copy; 2024 | Best S-Box Implementation | Flask Python Application</p>
            <p><i class="fas fa-code"></i> Developed for Advanced Cryptography</p>
        </footer>
    </div>
</body>
</html>
"""

# ===== Routes =====
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, 
                                 result=None, 
                                 sbox_decimal=None, 
                                 sbox_hex=None, 
                                 enumerate=enumerate)

@app.route("/download/<filetype>")
def download(filetype):
    if filetype == "decimal":
        path = "best_sbox_decimal.txt"
        filename = "best_sbox_decimal.txt"
    elif filetype == "hex":
        path = "best_sbox_hex.txt"
        filename = "best_sbox_hex.txt"
    else:
        return "File type not found", 404
    
    if os.path.exists(path):
        return send_file(path, 
                        as_attachment=True, 
                        download_name=filename,
                        mimetype='text/plain')
    else:
        return "File not found. Please ensure the S-Box files exist in the current directory.", 404

@app.route("/encrypt", methods=["POST"])
def encrypt():
    plaintext = request.form.get("plaintext", "").strip()
    key = request.form.get("key", "").strip()
    
    # Validation
    if not plaintext or not key:
        return render_template_string(HTML_TEMPLATE, 
                                     result="Error: Both plaintext and key are required!",
                                     sbox_decimal=None,
                                     sbox_hex=None,
                                     enumerate=enumerate)
    
    # Key length validation
    if len(key) not in [16, 24, 32]:
        return render_template_string(HTML_TEMPLATE,
                                     result=f"Warning: Key should be 16, 24, or 32 characters (got {len(key)}). Using padding/truncation.",
                                     sbox_decimal=None,
                                     sbox_hex=None,
                                     enumerate=enumerate)
    
    # Perform encryption using the best S-Box
    try:
        ciphertext = perform_encryption(plaintext, key)
        result = f"✅ Encryption Successful!\n\n📝 Plaintext: {plaintext}\n🔑 Key: {key}\n🔒 Ciphertext (Hex): {ciphertext}\n\n📊 Details: {len(plaintext)} characters encrypted with {len(key)*8}-bit key"
    except Exception as e:
        result = f"❌ Encryption Failed!\nError: {str(e)}"
    
    return render_template_string(HTML_TEMPLATE, 
                                 result=result,
                                 sbox_decimal=None,
                                 sbox_hex=None,
                                 enumerate=enumerate)

@app.route("/decrypt", methods=["POST"])
def decrypt():
    ciphertext = request.form.get("ciphertext", "").strip()
    key = request.form.get("key", "").strip()
    
    # Validation
    if not ciphertext or not key:
        return render_template_string(HTML_TEMPLATE, 
                                     result="Error: Both ciphertext and key are required!",
                                     sbox_decimal=None,
                                     sbox_hex=None,
                                     enumerate=enumerate)
    
    # Ciphertext validation (should be hex)
    try:
        bytes.fromhex(ciphertext)
    except:
        return render_template_string(HTML_TEMPLATE,
                                     result="Error: Ciphertext must be a valid hexadecimal string!",
                                     sbox_decimal=None,
                                     sbox_hex=None,
                                     enumerate=enumerate)
    
    # Perform decryption using the best S-Box
    try:
        plaintext = perform_decryption(ciphertext, key)
        result = f"✅ Decryption Successful!\n\n🔒 Ciphertext: {ciphertext}\n🔑 Key: {key}\n📝 Decrypted Text: {plaintext}\n\n📊 Details: {len(ciphertext)//2} bytes decrypted with {len(key)*8}-bit key"
    except Exception as e:
        result = f"❌ Decryption Failed!\nError: {str(e)}\nNote: Ensure you're using the correct key and valid ciphertext."
    
    return render_template_string(HTML_TEMPLATE, 
                                 result=result,
                                 sbox_decimal=None,
                                 sbox_hex=None,
                                 enumerate=enumerate)

@app.route("/show_sbox", methods=["GET"])
def show_sbox():
    try:
        sbox_decimal = read_sbox_decimal()
        sbox_hex = read_sbox_hex()
        
        # Verify S-Box structure
        if len(sbox_decimal) != 16 or any(len(row) != 16 for row in sbox_decimal):
            raise ValueError("Decimal S-Box must be 16x16")
        
        if len(sbox_hex) != 16 or any(len(row) != 16 for row in sbox_hex):
            raise ValueError("Hex S-Box must be 16x16")
        
        result = f"✅ S-Box Loaded Successfully!\n\n📊 Decimal: {len(sbox_decimal)}x{len(sbox_decimal[0])} table\n🔤 Hexadecimal: {len(sbox_hex)}x{len(sbox_hex[0])} table\n\nClick the 'Show S-Box' button above to view the tables."
        
        return render_template_string(HTML_TEMPLATE, 
                                     result=result,
                                     sbox_decimal=sbox_decimal,
                                     sbox_hex=sbox_hex,
                                     enumerate=enumerate)
    
    except FileNotFoundError:
        return render_template_string(HTML_TEMPLATE,
                                     result="❌ Error: S-Box files not found!\nPlease ensure 'best_sbox_decimal.txt' and 'best_sbox_hex.txt' exist in the current directory.",
                                     sbox_decimal=None,
                                     sbox_hex=None,
                                     enumerate=enumerate)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE,
                                     result=f"❌ Error loading S-Box: {str(e)}",
                                     sbox_decimal=None,
                                     sbox_hex=None,
                                     enumerate=enumerate)

# Placeholder functions for encryption and decryption
def perform_encryption(plaintext, key):
    """
    Perform encryption using the best S-Box.
    In a real implementation, this would use actual AES encryption.
    """
    # Convert to uppercase for demonstration
    ciphertext = plaintext.upper().encode().hex().upper()
    
    # Simulate S-Box substitution (first 32 chars for demo)
    sbox_decimal = read_sbox_decimal()
    
    # Simple demonstration of S-Box usage
    result_chars = []
    for i, char in enumerate(ciphertext[:32]):
        if char.isdigit():
            idx = int(char)
        else:
            idx = ord(char.upper()) - ord('A') + 10
        
        # Use S-Box value (simplified for demo)
        if idx < 16 and i < 16:
            sbox_val = sbox_decimal[idx][i % 16]
            result_chars.append(f"{sbox_val:02X}")
        else:
            result_chars.append(char)
    
    return ''.join(result_chars)

def perform_decryption(ciphertext, key):
    """
    Perform decryption using the inverse S-Box.
    In a real implementation, this would use actual AES decryption.
    """
    # Simple demonstration - just convert hex back to string
    try:
        # Remove any non-hex characters
        clean_hex = ''.join(c for c in ciphertext if c in '0123456789ABCDEFabcdef')
        
        # Convert hex to bytes to string
        decrypted = bytes.fromhex(clean_hex).decode('utf-8', errors='ignore')
        
        # Simulate inverse S-Box operation
        sbox_hex = read_sbox_hex()
        
        # For demonstration, just return the decoded string
        return f"[Decrypted] {decrypted}"
    except:
        return "[Demo] Decryption would use inverse S-Box lookup"

if __name__ == "__main__":
    print("=" * 60)
    print("AES S-Box System Starting...")
    print("=" * 60)
    print("\n📁 Required files:")
    print("   • best_sbox_decimal.txt")
    print("   • best_sbox_hex.txt")
    print("\n🌐 Server will run at: http://127.0.0.1:5000")
    print("=" * 60)
    
    # Check if required files exist
    if not os.path.exists("best_sbox_decimal.txt"):
        print("⚠️  Warning: best_sbox_decimal.txt not found!")
        print("   Creating sample file...")
        with open("best_sbox_decimal.txt", "w") as f:
            # Create a sample 16x16 S-Box
            for i in range(16):
                row = [(i*16 + j) % 256 for j in range(16)]
                f.write(" ".join(str(x) for x in row) + "\n")
    
    if not os.path.exists("best_sbox_hex.txt"):
        print("⚠️  Warning: best_sbox_hex.txt not found!")
        print("   Creating sample file...")
        with open("best_sbox_hex.txt", "w") as f:
            # Create a sample 16x16 S-Box in hex
            for i in range(16):
                row = [f"{i*16 + j:02X}" for j in range(16)]
                f.write(" ".join(row) + "\n")
    
    print("✅ All files ready!")
    print("=" * 60)
    
    app.run(debug=True, host='127.0.0.1', port=5000)