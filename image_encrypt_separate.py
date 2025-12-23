import numpy as np
import streamlit as st
from PIL import Image
import io

def encrypt_image_new(image, sbox, key_str=""):
    """Encrypt image with S-box"""
    import hashlib
    
    if key_str:
        key = hashlib.md5(key_str.encode()).digest()
    else:
        key = bytes([0] * 16)
    
    key_np = np.frombuffer(key, dtype=np.uint8)
    
    img_np = np.array(image, dtype=np.uint8)
    original_shape = img_np.shape
    flat_img = img_np.flatten()
    
    repeated_key = np.tile(key_np, len(flat_img) // len(key_np) + 1)[:len(flat_img)]
    xored = flat_img ^ repeated_key
    
    if len(sbox.shape) == 2:
        sbox_flat = sbox.flatten()
    else:
        sbox_flat = sbox
    
    encrypted_flat = np.array([sbox_flat[b] for b in xored], dtype=np.uint8)
    
    return encrypted_flat.reshape(original_shape)


def decrypt_image_new(encrypted_np, sbox, key_str=""):
    """Decrypt image with S-box"""
    import hashlib
    
    if key_str:
        key = hashlib.md5(key_str.encode()).digest()
    else:
        key = bytes([0] * 16)
    
    key_np = np.frombuffer(key, dtype=np.uint8)
    
    original_shape = encrypted_np.shape
    flat_enc = encrypted_np.flatten()
    
    # Create inverse S-box
    if len(sbox.shape) == 2:
        sbox_flat = sbox.flatten()
    else:
        sbox_flat = sbox
    
    inv_sbox = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        inv_sbox[sbox_flat[i]] = i
    
    decrypted_flat = np.array([inv_sbox[b] for b in flat_enc], dtype=np.uint8)
    
    repeated_key = np.tile(key_np, len(decrypted_flat) // len(key_np) + 1)[:len(decrypted_flat)]
    final_flat = decrypted_flat ^ repeated_key
    
    return final_flat.reshape(original_shape)


def image_entropy(img):
    """Calculate Shannon entropy"""
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0,255), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def npcr_uaci(img1, img2):
    """Calculate NPCR and UACI"""
    diff = img1 != img2
    npcr = np.sum(diff) / diff.size * 100
    uaci = np.mean(np.abs(img1.astype(np.int16) - img2.astype(np.int16)) / 255) * 100
    return float(npcr), float(uaci)


def validate_pixel_perfect(original, decrypted):
    """Check if decryption is perfect"""
    return bool(np.array_equal(original, decrypted))


def image_encrypt_ui_new(active_sbox, sbox_choice):
    """UI untuk enkripsi/dekripsi gambar dengan tombol independen"""
    st.markdown("---")
    st.markdown("""
    <style>
    /* Image Section Header */
    .image-section {
        padding: 25px;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 25px;
        animation: fadeIn 0.6s ease-in;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    /* Card Styling */
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        animation: slideUp 0.5s ease-out;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid rgba(102, 126, 234, 0.2);
    }
    
    .encrypted-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    }
    
    .decrypted-card {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.8; }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 10px rgba(102, 126, 234, 0.5); }
        50% { box-shadow: 0 0 25px rgba(102, 126, 234, 0.8); }
    }
    
    /* Metric Box */
    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        animation: slideIn 0.5s ease-out;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    /* Success Badge */
    .success-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: bold;
        display: inline-block;
        animation: pulse 2s infinite;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .warning-badge {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: bold;
        display: inline-block;
        animation: pulse 2s infinite;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
    }
    
    /* Button Styling */
    .stButton>button {
        transition: all 0.3s ease;
        border-radius: 12px;
        font-weight: 600;
        border: none;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        animation: glow 1.5s infinite;
    }
    
    /* Image Container */
    .img-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        animation: fadeIn 0.8s ease-out;
    }
    
    .img-container:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 35px rgba(0,0,0,0.2);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.2);
        border-radius: 8px;
        padding: 12px 24px;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255,255,255,0.3);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #667eea !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="image-section"><h2 style="margin:0;">🖼️ Image Encryption / Decryption</h2><p style="margin:5px 0 0 0; opacity: 0.9;">Encrypt and decrypt images using S-box substitution</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        uploaded = st.file_uploader(
            "📤 Upload Image (PNG / JPG / JPEG)",
            type=["png","jpg","jpeg"],
            key="image_encrypt_file",
            help="Upload an image to encrypt or decrypt"
        )
    
    with col2:
        key_input = st.text_input(
            "🔑 Encryption/Decryption Key", 
            value="my-secret-key",
            type="password",
            key="image_key",
            help="Enter a key for encryption/decryption (must be same for encrypt & decrypt)"
        )

    if uploaded is None or active_sbox is None:
        st.info("👆 Please upload an image and select an S-box to continue")
        return

    try:
        # Load image
        img = Image.open(uploaded).convert("RGB")
        img_np = np.array(img, dtype=np.uint8)
        
        # Store original image in session
        st.session_state.current_original_image = img_np
        
        # Info box with gradient
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%); 
                    padding: 15px; border-radius: 10px; margin: 15px 0;
                    border-left: 4px solid #0ea5e9; animation: slideIn 0.5s ease-out;
                    color: #0c4a6e;">
            <strong style="color: #0c4a6e;">✅ Image loaded successfully!</strong><br>
            <span style="color: #075985;">📐 Size: {img_np.shape[1]} x {img_np.shape[0]} pixels ({img_np.shape[2]} channels)</span><br>
            <span style="color: #075985;">🔢 S-box: <strong style="color: #0c4a6e;">{sbox_choice}</strong></span>
        </div>
        """, unsafe_allow_html=True)

        # Original Image Display
        st.markdown("### 📷 Original Image")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(img_np)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Action Buttons in Tabs
        tab1, tab2, tab3 = st.tabs(["🔒 Encrypt", "🔓 Decrypt", "🗑️ Clear"])
        
        with tab1:
            st.markdown("### Encrypt Image")
            st.write("Transform your image into an encrypted format using the selected S-box.")
            
            if st.button("🔒 Start Encryption", type="primary", key="encrypt_image_btn"):
                with st.spinner("🔄 Encrypting image... Please wait"):
                    encrypted = encrypt_image_new(img_np, active_sbox, key_input)
                    st.session_state.encrypted_image = encrypted
                    st.session_state.show_encrypted_image = True
                    st.session_state.original_image = img_np
                    st.session_state.stored_image_key = key_input
                    st.session_state.image_sbox_name = sbox_choice
                    st.session_state.has_original_for_validation = True
                st.success("✅ Image encrypted successfully!")
                st.balloons()
                st.rerun()
        
        with tab2:
            st.markdown("### Decrypt Image")
            has_encrypted_in_session = (st.session_state.encrypted_image is not None and 
                                       st.session_state.show_encrypted_image)
            
            if has_encrypted_in_session:
                st.write("Decrypt the encrypted image from current session.")
            else:
                st.write("Decrypt the uploaded image (treat as ciphertext).")
            
            if st.button("🔓 Start Decryption", type="primary", key="decrypt_image_btn"):
                # Tentukan source untuk decrypt
                if has_encrypted_in_session:
                    source_for_decrypt = st.session_state.encrypted_image
                    decrypt_from_session = True
                    st.info("🔄 Decrypting from encrypted image in session...")
                else:
                    source_for_decrypt = img_np
                    decrypt_from_session = False
                
                with st.spinner("🔄 Decrypting image... Please wait"):
                    decrypted = decrypt_image_new(source_for_decrypt, active_sbox, key_input)
                    st.session_state.decrypted_image = decrypted
                    st.session_state.show_decrypted_image = True
                    st.session_state.decrypt_source_image = source_for_decrypt
                    st.session_state.decrypt_from_current_session = decrypt_from_session
                
                st.success("✅ Image decrypted successfully!")
                st.balloons()
                st.rerun()
        
        with tab3:
            st.markdown("### Clear All Results")
            st.write("Remove all encrypted and decrypted images from the session.")
            
            if st.button("🗑️ Clear Everything", key="clear_all_images"):
                st.session_state.encrypted_image = None
                st.session_state.decrypted_image = None
                st.session_state.show_encrypted_image = False
                st.session_state.show_decrypted_image = False
                st.session_state.original_image = None
                st.session_state.has_original_for_validation = False
                st.session_state.decrypt_from_current_session = True
                st.session_state.decrypt_source_image = None
                st.success("🧹 All results cleared!")
                st.rerun()

        # Display encrypted image in a clean card
        if st.session_state.show_encrypted_image and st.session_state.encrypted_image is not None:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown('<div class="result-card encrypted-card">', unsafe_allow_html=True)
            st.markdown("### 🔒 Encrypted Result")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown('<div class="img-container">', unsafe_allow_html=True)
                st.image(st.session_state.encrypted_image)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.markdown(f"**🎯 S-box:** `{st.session_state.get('image_sbox_name', sbox_choice)}`")
                st.markdown(f"**🔑 Key:** `{'*' * min(len(st.session_state.get('stored_image_key', '')), 16)}...`")
                st.markdown("**📊 Status:** <span class='success-badge'>🔒 Encrypted</span>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Show metrics for encrypted image
            if 'original_image' in st.session_state:
                st.markdown("<br>", unsafe_allow_html=True)
                entropy_original = image_entropy(st.session_state.original_image)
                entropy_encrypted = image_entropy(st.session_state.encrypted_image)
                npcr, uaci = npcr_uaci(st.session_state.original_image, st.session_state.encrypted_image)
                
                st.markdown("#### 📊 Encryption Quality Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Entropy (Original)", f"{entropy_original:.4f}", 
                             help="Randomness of original image")
                
                with col2:
                    delta_entropy = entropy_encrypted - entropy_original
                    st.metric("Entropy (Encrypted)", f"{entropy_encrypted:.4f}", 
                             delta=f"{delta_entropy:+.4f}",
                             help="Higher is better - indicates randomness")
                
                with col3:
                    st.metric("NPCR", f"{npcr:.2f}%", help="Pixel change rate (ideal: ~99.6%)")
                    ideal_npcr = 99.6094
                    if abs(npcr - ideal_npcr) < 0.5:
                        st.markdown("<span class='success-badge'>✅ Excellent</span>", unsafe_allow_html=True)
                    elif abs(npcr - ideal_npcr) < 1.0:
                        st.markdown("<span class='warning-badge'>👍 Good</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='warning-badge'>⚠️ Moderate</span>", unsafe_allow_html=True)
                
                with col4:
                    st.metric("UACI", f"{uaci:.2f}%", help="Intensity change (ideal: ~33.46%)")
                    ideal_uaci = 33.4635
                    if abs(uaci - ideal_uaci) < 0.5:
                        st.markdown("<span class='success-badge'>✅ Excellent</span>", unsafe_allow_html=True)
                    elif abs(uaci - ideal_uaci) < 1.0:
                        st.markdown("<span class='warning-badge'>👍 Good</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='warning-badge'>⚠️ Moderate</span>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Display decrypted image in a clean card
        if st.session_state.show_decrypted_image and st.session_state.decrypted_image is not None:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown('<div class="result-card decrypted-card">', unsafe_allow_html=True)
            st.markdown("### 🔓 Decrypted Result")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown('<div class="img-container">', unsafe_allow_html=True)
                st.image(st.session_state.decrypted_image)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.markdown(f"**🎯 S-box:** `{sbox_choice}`")
                st.markdown(f"**🔑 Key Used:** `{'*' * min(len(key_input), 16)}...`")
                
                # Check if we can validate
                can_validate = False
                if (st.session_state.get('has_original_for_validation', False) and 
                    st.session_state.encrypted_image is not None and
                    st.session_state.get('decrypt_source_image') is not None):
                    if np.array_equal(st.session_state.encrypted_image, st.session_state.decrypt_source_image):
                        can_validate = True
                
                if can_validate and 'original_image' in st.session_state and st.session_state.original_image is not None:
                    is_perfect = validate_pixel_perfect(
                        st.session_state.original_image, 
                        st.session_state.decrypted_image
                    )
                    if is_perfect:
                        st.markdown("**📊 Status:** <span class='success-badge'>✅ Perfect Match</span>", unsafe_allow_html=True)
                        st.success("🎉 Decryption successful! 100% match with original.")
                    else:
                        st.markdown("**📊 Status:** <span class='warning-badge'>⚠️ Mismatch</span>", unsafe_allow_html=True)
                        st.warning("❌ Result differs from original. Check key/S-box.")
                else:
                    st.markdown("**📊 Status:** <span class='success-badge'>✅ Decrypted</span>", unsafe_allow_html=True)
                    st.info("✓ Decryption completed successfully.")
                    if st.session_state.get('decrypt_from_current_session', True) == False:
                        st.caption("💡 Tip: Encrypt an image first, then decrypt to validate.")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        with st.expander("🔍 Show Error Details"):
            import traceback
            st.code(traceback.format_exc())