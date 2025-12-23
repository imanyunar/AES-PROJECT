import io
import json
import pandas as pd
import streamlit as st

from crypto_functions import encrypt_image, decrypt_image
from image_crypto import (
    load_image,
    calculate_entropy,
    calculate_npcr,
    calculate_uaci,
    histogram_analysis,
    test_image_sboxes
)

# ============================================================
# IMAGE SBOX REGISTRY
# ============================================================

def extend_sbox_registry(sboxes: dict) -> dict:
    image_sboxes = test_image_sboxes()
    for name, sbox in image_sboxes.items():
        sboxes[f"IMG-{name}"] = sbox
    return sboxes


# ============================================================
# IMAGE ENCRYPTION UI
# ============================================================

def image_encrypt_ui(active_sbox, sbox_choice):
    st.markdown("---")
    st.subheader("🖼️ Image Encryption & Decryption")

    uploaded = st.file_uploader(
        "Upload Image (PNG / JPG)",
        type=["png", "jpg", "jpeg"],
        key="image_encrypt"
    )

    if not uploaded or active_sbox is None:
        return

    image = load_image(uploaded)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image")

    encrypted = encrypt_image(image, active_sbox)

    with col2:
        st.image(encrypted, caption="Encrypted Image")

    decrypted = decrypt_image(encrypted, active_sbox)
    st.image(decrypted, caption="Decrypted Image")

    # ================= Metrics =================
    st.subheader("📊 Image Encryption Metrics")

    metrics = {
        "S-Box": sbox_choice,
        "Entropy Plain": calculate_entropy(image),
        "Entropy Cipher": calculate_entropy(encrypted),
        "NPCR (%)": calculate_npcr(image, encrypted),
        "UACI (%)": calculate_uaci(image, encrypted)
    }

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entropy Plain", f"{metrics['Entropy Plain']:.4f}")
    c2.metric("Entropy Cipher", f"{metrics['Entropy Cipher']:.4f}")
    c3.metric("NPCR (%)", f"{metrics['NPCR (%)']:.4f}")
    c4.metric("UACI (%)", f"{metrics['UACI (%)']:.4f}")

    # ================= Export =================
    st.subheader("⬇️ Export Image Metrics")

    df = pd.DataFrame([metrics])

    st.download_button(
        "Download JSON",
        json.dumps(metrics, indent=4),
        file_name="image_metrics.json",
        mime="application/json"
    )

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)

    st.download_button(
        "Download CSV",
        csv_buf.getvalue(),
        file_name="image_metrics.csv",
        mime="text/csv"
    )


# ============================================================
# IMAGE SBOX COMPARISON PAGE
# ============================================================

def image_comparison_page():
    st.header("🖼️ Image S-Box Comparison")

    uploaded = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"],
        key="image_compare"
    )

    if not uploaded:
        return

    image = load_image(uploaded)
    results = []

    for name, sbox in test_image_sboxes().items():
        encrypted = encrypt_image(image, sbox)

        results.append({
            "S-Box": name,
            "Entropy Cipher": calculate_entropy(encrypted),
            "NPCR (%)": calculate_npcr(image, encrypted),
            "UACI (%)": calculate_uaci(image, encrypted),
            "Histogram Bins": sum(histogram_analysis(encrypted).values())
        })

    df = pd.DataFrame(results)

    st.dataframe(df)

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)

    st.download_button(
        "⬇️ Download Comparison CSV",
        csv_buf.getvalue(),
        file_name="image_sbox_comparison.csv",
        mime="text/csv"
    )
