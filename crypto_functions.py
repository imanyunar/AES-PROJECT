import numpy as np
import streamlit as st
from PIL import Image
import io
import pandas as pd
import matplotlib.pyplot as plt

def create_inverse_sbox(sbox):
    """Create proper inverse S-box"""
    # Handle both 1D and 2D S-box
    if len(sbox.shape) == 2:
        sbox_flat = sbox.flatten()
    else:
        sbox_flat = sbox
    
    # Create inverse mapping
    inv_sbox = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        inv_sbox[sbox_flat[i]] = i
    
    return inv_sbox

def encrypt_image(image, sbox, key_str=""):
    """
    Encrypt image: (Pixel XOR Key) then S-box substitution
    Process: plaintext -> XOR with key -> substitute with S-box -> ciphertext
    """
    import hashlib
    
    # Generate key dari string
    if key_str:
        key = hashlib.md5(key_str.encode()).digest()
    else:
        key = bytes([0] * 16)
    
    key_np = np.frombuffer(key, dtype=np.uint8)
    
    # Convert image to numpy array
    img_np = np.array(image, dtype=np.uint8)
    original_shape = img_np.shape
    flat_img = img_np.flatten()
    
    # Repeat key to match image length
    repeated_key = np.tile(key_np, len(flat_img) // len(key_np) + 1)[:len(flat_img)]
    
    # Step 1: XOR with key
    xored = flat_img ^ repeated_key
    
    # Step 2: S-box substitution
    if len(sbox.shape) == 2:
        sbox_flat = sbox.flatten()
    else:
        sbox_flat = sbox
    
    encrypted_flat = np.array([sbox_flat[b] for b in xored], dtype=np.uint8)
    
    return encrypted_flat.reshape(original_shape)

def decrypt_image(encrypted_np, sbox, key_str=""):
    """
    Decrypt image: Inverse S-box substitution then XOR with Key
    Process: ciphertext -> inverse substitute -> XOR with key -> plaintext
    """
    import hashlib
    
    # Generate key dari string (must be same as encryption)
    if key_str:
        key = hashlib.md5(key_str.encode()).digest()
    else:
        key = bytes([0] * 16)
    
    key_np = np.frombuffer(key, dtype=np.uint8)
    
    # Get original shape
    original_shape = encrypted_np.shape
    flat_enc = encrypted_np.flatten()
    
    # Create inverse S-box
    inv_sbox = create_inverse_sbox(sbox)
    
    # Step 1: Inverse S-box substitution
    decrypted_flat = np.array([inv_sbox[b] for b in flat_enc], dtype=np.uint8)
    
    # Step 2: XOR with key (reverse the XOR operation)
    repeated_key = np.tile(key_np, len(decrypted_flat) // len(key_np) + 1)[:len(decrypted_flat)]
    final_flat = decrypted_flat ^ repeated_key
    
    return final_flat.reshape(original_shape)

def image_entropy(img):
    """Calculate Shannon entropy of image"""
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0,255), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

def npcr_uaci(img1, img2):
    """Calculate NPCR and UACI metrics"""
    diff = img1 != img2
    npcr = np.sum(diff) / diff.size * 100
    uaci = np.mean(np.abs(img1.astype(np.int16) - img2.astype(np.int16)) / 255) * 100
    return float(npcr), float(uaci)

def histogram_bins(img):
    """Count number of bins with non-zero values"""
    if img.ndim == 3:
        bins = []
        for c in range(3):
            h = np.histogram(img[:,:,c], bins=256, range=(0,255))[0]
            bins.append(np.count_nonzero(h))
        return int(sum(bins) // 3)
    h = np.histogram(img, bins=256, range=(0,255))[0]
    return int(np.count_nonzero(h))

def plot_rgb_histogram(img, title):
    """Plot histogram for RGB or grayscale image"""
    fig, ax = plt.subplots(figsize=(8, 4))
    if img.ndim == 3:
        colors = ['red', 'green', 'blue']
        for i, col in enumerate(colors):
            hist = np.histogram(img[:,:,i], bins=256, range=(0,255))[0]
            ax.plot(hist, color=col, label=col.upper(), alpha=0.7)
    else:
        hist = np.histogram(img, bins=256, range=(0,255))[0]
        ax.plot(hist, color='gray', label='Grayscale')
    ax.set_title(title)
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.set_xlim(0, 256)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

def validate_pixel_perfect(original, decrypted):
    """Check if decryption is pixel-perfect"""
    return bool(np.array_equal(original, decrypted))

def image_encrypt_ui(active_sbox, sbox_choice):
    """UI untuk enkripsi/dekripsi gambar dengan key"""
    st.markdown("---")
    st.subheader("🖼️ Image Encryption / Decryption")

    col1, col2 = st.columns(2)
    
    with col1:
        uploaded = st.file_uploader(
            "Upload Image (PNG / JPG / JPEG)",
            type=["png","jpg","jpeg"],
            key="image_encrypt_file"
        )
    
    with col2:
        key_input = st.text_input(
            "Encryption Key", 
            value="my-secret-key",
            type="password",
            key="image_key",
            help="Enter a key for encryption/decryption"
        )

    if uploaded is None or active_sbox is None:
        st.info("👆 Please upload an image and select an S-box to continue")
        return

    try:
        # Load image
        img = Image.open(uploaded).convert("RGB")
        img_np = np.array(img, dtype=np.uint8)
        
        st.info(f"📐 Image size: {img_np.shape[1]} x {img_np.shape[0]} pixels ({img_np.shape[2]} channels)")

        # Encrypt
        with st.spinner("Encrypting..."):
            encrypted = encrypt_image(img_np, active_sbox, key_input)
        
        # Decrypt
        with st.spinner("Decrypting..."):
            decrypted = decrypt_image(encrypted, active_sbox, key_input)

        # Display images
        st.markdown("### 🖼️ Image Comparison")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(img_np, caption="📷 Original")
            st.caption(f"Size: {img_np.shape[1]}x{img_np.shape[0]}")
        
        with col2:
            st.image(encrypted, caption="🔒 Encrypted")
            st.caption(f"S-box: {sbox_choice}")
        
        with col3:
            st.image(decrypted, caption="🔓 Decrypted")
            # Check if decryption is perfect
            is_perfect = validate_pixel_perfect(img_np, decrypted)
            if is_perfect:
                st.caption("✅ Perfect match!")
            else:
                st.caption("⚠️ Mismatch detected")

        # Calculate metrics
        entropy_original = image_entropy(img_np)
        entropy_encrypted = image_entropy(encrypted)
        npcr, uaci = npcr_uaci(img_np, encrypted)
        bins_original = histogram_bins(img_np)
        bins_encrypted = histogram_bins(encrypted)
        is_perfect = validate_pixel_perfect(img_np, decrypted)

        # Display metrics
        st.markdown("### 📊 Image Encryption Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Entropy (Original)", 
                f"{entropy_original:.4f}",
                help="Shannon entropy of original image"
            )
            st.metric(
                "Entropy (Encrypted)", 
                f"{entropy_encrypted:.4f}",
                delta=f"+{entropy_encrypted - entropy_original:.4f}",
                help="Higher entropy = more random"
            )
        
        with col2:
            st.metric(
                "NPCR", 
                f"{npcr:.2f}%",
                help="Number of Pixels Change Rate (ideal: ~99.6%)"
            )
            ideal_npcr = 99.6094
            if abs(npcr - ideal_npcr) < 0.5:
                st.success("✅ Excellent")
            elif abs(npcr - ideal_npcr) < 1.0:
                st.info("👍 Good")
            else:
                st.warning("⚠️ Moderate")
        
        with col3:
            st.metric(
                "UACI", 
                f"{uaci:.2f}%",
                help="Unified Average Changing Intensity (ideal: ~33.46%)"
            )
            ideal_uaci = 33.4635
            if abs(uaci - ideal_uaci) < 0.5:
                st.success("✅ Excellent")
            elif abs(uaci - ideal_uaci) < 1.0:
                st.info("👍 Good")
            else:
                st.warning("⚠️ Moderate")
        
        with col4:
            st.metric(
                "Histogram Bins (Original)", 
                bins_original,
                help="Number of unique pixel values"
            )
            st.metric(
                "Histogram Bins (Encrypted)", 
                bins_encrypted,
                delta=f"+{bins_encrypted - bins_original}",
                help="Ideal: 256 (uniform distribution)"
            )
        
        # Decryption validation
        st.markdown("### 🔍 Decryption Validation")
        if is_perfect:
            st.success("✅ **Pixel-Perfect Decryption**: All pixels match exactly! Encryption is reversible.")
            st.metric("Error Rate", "0.000%")
        else:
            error_pixels = np.sum(img_np != decrypted)
            total_pixels = img_np.size
            error_rate = (error_pixels / total_pixels) * 100
            st.error(f"❌ **Decryption Failed**: {error_pixels:,} pixels don't match ({error_rate:.3f}% error)")
            
            # Show first few mismatches
            if error_pixels > 0:
                mismatch_indices = np.where(img_np.flatten() != decrypted.flatten())[0]
                st.write(f"First mismatches at indices: {mismatch_indices[:10].tolist()}")

        # Histogram visualization
        with st.expander("📊 Show Detailed Histograms"):
            col1, col2 = st.columns(2)
            with col1:
                plot_rgb_histogram(img_np, "Original Image Histogram")
            with col2:
                plot_rgb_histogram(encrypted, "Encrypted Image Histogram")

        # Download encrypted image
        st.markdown("### ⬇️ Download Encrypted Image")
        col1, col2 = st.columns(2)
        
        with col1:
            # Convert to PIL Image for download
            encrypted_pil = Image.fromarray(encrypted)
            buf = io.BytesIO()
            encrypted_pil.save(buf, format='PNG')
            
            st.download_button(
                label="📥 Download Encrypted PNG",
                data=buf.getvalue(),
                file_name=f"encrypted_{sbox_choice}.png",
                mime="image/png"
            )
        
        with col2:
            # Download decrypted image
            decrypted_pil = Image.fromarray(decrypted)
            buf2 = io.BytesIO()
            decrypted_pil.save(buf2, format='PNG')
            
            st.download_button(
                label="📥 Download Decrypted PNG",
                data=buf2.getvalue(),
                file_name=f"decrypted_{sbox_choice}.png",
                mime="image/png"
            )

        # Save metrics to session state
        if "image_metrics" not in st.session_state:
            st.session_state.image_metrics = []

        metric_data = {
            "S-Box": sbox_choice,
            "Entropy_Original": round(entropy_original, 6),
            "Entropy_Encrypted": round(entropy_encrypted, 6),
            "NPCR (%)": round(npcr, 4),
            "UACI (%)": round(uaci, 4),
            "Histogram_Bins_Original": bins_original,
            "Histogram_Bins_Encrypted": bins_encrypted,
            "PixelPerfect": "Yes" if is_perfect else "No",
            "Error_Rate": 0.0 if is_perfect else error_rate
        }
        
        # Update or add to metrics list
        existing_idx = None
        for idx, m in enumerate(st.session_state.image_metrics):
            if m["S-Box"] == sbox_choice:
                existing_idx = idx
                break
        
        if existing_idx is not None:
            st.session_state.image_metrics[existing_idx] = metric_data
        else:
            st.session_state.image_metrics.append(metric_data)

    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def image_comparison_page(sboxes):
    """Page untuk membandingkan S-boxes pada gambar"""
    st.header("🖼️ Image S-Box Comparison")
    st.caption("Compare image encryption metrics across different S-boxes")

    col1, col2 = st.columns(2)
    
    with col1:
        uploaded = st.file_uploader(
            "Upload Image for Comparison",
            type=["png", "jpg", "jpeg"],
            key="image_comparison_file"
        )
    
    with col2:
        key_input = st.text_input(
            "Encryption Key", 
            value="comparison-key",
            type="password",
            key="image_comparison_key",
            help="Use the same key for fair comparison across S-boxes"
        )

    if uploaded is None:
        st.info("👆 Please upload an image to start comparison")
        return

    try:
        # Load image
        img = Image.open(uploaded).convert("RGB")
        img_np = np.array(img, dtype=np.uint8)
        
        st.success(f"✅ Image loaded: {img_np.shape[1]} x {img_np.shape[0]} pixels")

        # Categorize S-boxes
        available_sboxes = list(sboxes.keys())
        text_sboxes = [name for name in available_sboxes if not name.startswith("IMG-")]
        image_sboxes = [name for name in available_sboxes if name.startswith("IMG-")]
        
        st.markdown("### 🎯 Select S-Boxes to Compare")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📝 Text S-Boxes (Affine Matrix)**")
            selected_text = st.multiselect(
                "Text S-Boxes",
                options=text_sboxes,
                default=text_sboxes[:2] if len(text_sboxes) >= 2 else text_sboxes,
                key="text_sbox_select",
                label_visibility="collapsed"
            )
            st.caption(f"{len(selected_text)} selected")
        
        with col2:
            st.markdown("**🖼️ Image S-Boxes (Alamsyah et al.)**")
            selected_image = st.multiselect(
                "Image S-Boxes",
                options=image_sboxes,
                default=image_sboxes if len(image_sboxes) <= 3 else image_sboxes[:3],
                key="image_sbox_select",
                label_visibility="collapsed"
            )
            st.caption(f"{len(selected_image)} selected")
        
        # Combine selections
        selected_sboxes = selected_text + selected_image
        
        # Quick select buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("✅ Select All"):
                st.session_state.text_sbox_select = text_sboxes
                st.session_state.image_sbox_select = image_sboxes
                st.rerun()
        with col2:
            if st.button("📝 Text Only"):
                st.session_state.text_sbox_select = text_sboxes
                st.session_state.image_sbox_select = []
                st.rerun()
        with col3:
            if st.button("🖼️ Image Only"):
                st.session_state.text_sbox_select = []
                st.session_state.image_sbox_select = image_sboxes
                st.rerun()
        with col4:
            if st.button("❌ Clear All"):
                st.session_state.text_sbox_select = []
                st.session_state.image_sbox_select = []
                st.rerun()

        if not selected_sboxes:
            st.warning("⚠️ Please select at least one S-box to compare")
            return
        
        st.info(f"📊 Ready to compare **{len(selected_sboxes)}** S-boxes: {len(selected_text)} text + {len(selected_image)} image")

        if st.button("🔬 Run Comparison", type="primary"):
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, sbox_name in enumerate(selected_sboxes):
                status_text.text(f"⏳ Processing {sbox_name}... ({idx+1}/{len(selected_sboxes)})")
                
                sbox = sboxes[sbox_name]
                
                # Encrypt and decrypt
                encrypted = encrypt_image(img_np, sbox, key_input)
                decrypted = decrypt_image(encrypted, sbox, key_input)
                
                # Calculate metrics
                entropy_encrypted = image_entropy(encrypted)
                npcr, uaci = npcr_uaci(img_np, encrypted)
                bins = histogram_bins(encrypted)
                is_perfect = validate_pixel_perfect(img_np, decrypted)
                
                # Calculate error rate
                if is_perfect:
                    error_rate = 0.0
                else:
                    error_pixels = np.sum(img_np != decrypted)
                    error_rate = (error_pixels / img_np.size) * 100
                
                # Categorize S-box type
                sbox_type = "🖼️ Image" if sbox_name.startswith("IMG-") else "📝 Text"
                
                results.append({
                    "S-Box": sbox_name,
                    "Type": sbox_type,
                    "Entropy": round(entropy_encrypted, 6),
                    "NPCR (%)": round(npcr, 4),
                    "UACI (%)": round(uaci, 4),
                    "Histogram_Bins": bins,
                    "Reversible": "✅ Yes" if is_perfect else "❌ No",
                    "Error_Rate (%)": round(error_rate, 6)
                })
                
                progress_bar.progress((idx + 1) / len(selected_sboxes))
            
            status_text.empty()
            progress_bar.empty()
            
            st.success(f"✅ Comparison completed for {len(selected_sboxes)} S-boxes!")

            # Display results
            st.markdown("### 📊 Comparison Results")
            df = pd.DataFrame(results)
            st.dataframe(df)

            # Separate analysis by type
            if len(selected_text) > 0 and len(selected_image) > 0:
                st.markdown("### 🔍 Analysis by S-Box Type")
                
                tab1, tab2, tab3 = st.tabs(["📝 Text S-Boxes", "🖼️ Image S-Boxes", "⚖️ Comparison"])
                
                with tab1:
                    text_df = df[df['Type'] == '📝 Text']
                    if len(text_df) > 0:
                        st.dataframe(text_df)
                        st.caption(f"Average Entropy: {text_df['Entropy'].mean():.4f} | Average NPCR: {text_df['NPCR (%)'].mean():.2f}%")
                    else:
                        st.info("No text S-boxes selected")
                
                with tab2:
                    image_df = df[df['Type'] == '🖼️ Image']
                    if len(image_df) > 0:
                        st.dataframe(image_df)
                        st.caption(f"Average Entropy: {image_df['Entropy'].mean():.4f} | Average NPCR: {image_df['NPCR (%)'].mean():.2f}%")
                    else:
                        st.info("No image S-boxes selected")
                
                with tab3:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📝 Text S-Boxes Performance**")
                        if len(text_df) > 0:
                            st.metric("Avg Entropy", f"{text_df['Entropy'].mean():.4f}")
                            st.metric("Avg NPCR", f"{text_df['NPCR (%)'].mean():.2f}%")
                            st.metric("Avg UACI", f"{text_df['UACI (%)'].mean():.2f}%")
                        else:
                            st.info("No data")
                    
                    with col2:
                        st.markdown("**🖼️ Image S-Boxes Performance**")
                        if len(image_df) > 0:
                            st.metric("Avg Entropy", f"{image_df['Entropy'].mean():.4f}")
                            st.metric("Avg NPCR", f"{image_df['NPCR (%)'].mean():.2f}%")
                            st.metric("Avg UACI", f"{image_df['UACI (%)'].mean():.2f}%")
                        else:
                            st.info("No data")

            # Statistics summary
            st.markdown("### 📈 Statistical Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Entropy", f"{df['Entropy'].mean():.4f}")
                st.caption(f"Range: {df['Entropy'].min():.4f} - {df['Entropy'].max():.4f}")
            
            with col2:
                st.metric("Average NPCR", f"{df['NPCR (%)'].mean():.2f}%")
                st.caption(f"Ideal: ~99.61%")
            
            with col3:
                st.metric("Average UACI", f"{df['UACI (%)'].mean():.2f}%")
                st.caption(f"Ideal: ~33.46%")

            # Visualizations
            st.markdown("### 📊 Visual Comparisons")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Entropy", "NPCR", "UACI", "Histogram Bins"])
            
            with tab1:
                st.bar_chart(df.set_index("S-Box")["Entropy"])
                st.caption("Higher entropy indicates better randomness (ideal: ~8.0 for 8-bit data)")
            
            with tab2:
                st.bar_chart(df.set_index("S-Box")["NPCR (%)"])
                st.caption("Higher NPCR is better (ideal: ~99.6094%)")
            
            with tab3:
                st.bar_chart(df.set_index("S-Box")["UACI (%)"])
                st.caption("Closer to 33.46% is better")
            
            with tab4:
                st.bar_chart(df.set_index("S-Box")["Histogram_Bins"])
                st.caption("Higher bins = more uniform distribution (ideal: 256)")

            # Check reversibility
            reversible_count = sum(1 for r in results if r["Reversible"] == "✅ Yes")
            if reversible_count == len(results):
                st.success(f"🎉 All {len(results)} S-boxes are perfectly reversible!")
            else:
                st.warning(f"⚠️ Only {reversible_count}/{len(results)} S-boxes are reversible")

            # Download options
            st.markdown("### ⬇️ Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📄 Download CSV",
                    csv_data,
                    file_name=f"image_sbox_comparison_{len(selected_sboxes)}_sboxes.csv",
                    mime="text/csv"
                )
            
            with col2:
                import json
                json_data = json.dumps(results, indent=2)
                st.download_button(
                    "📋 Download JSON",
                    json_data,
                    file_name=f"image_sbox_comparison_{len(selected_sboxes)}_sboxes.json",
                    mime="application/json"
                )

    except Exception as e:
        st.error(f"❌ Error during comparison: {str(e)}")
        import traceback
        st.code(traceback.format_exc())