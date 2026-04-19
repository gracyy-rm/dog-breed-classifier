
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
from PIL import Image
from src.inference import transform, device, model, le



# ---------------- Page Config ----------------//
st.set_page_config(
    page_title="Dog Breed AI",
    page_icon="🐶",
    layout="wide"
)

# ---------------- Background Gradient ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.title("📊 Model Info")
st.sidebar.write("🐶 Dog Breed Classifier")
st.sidebar.write("📦 Model: ResNet50")
st.sidebar.write("🎯 Classes: 120 dog breeds")
st.sidebar.write("📈 Accuracy: ~85%")

st.sidebar.markdown("---")
st.sidebar.info("Upload an image to get predictions")

# ---------------- Header ----------------
st.markdown("<h1 style='text-align:center;'>🐶 Dog Breed Classifier AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Top-3 Breed Prediction with Confidence Scores</p>", unsafe_allow_html=True)

# ---------------- Upload ----------------
uploaded_file = st.file_uploader("Upload a Dog Image", type=["jpg", "png", "jpeg"])

# ---------------- Prediction Function ----------------
def predict_topk(image_file, k=3):
    image = Image.open(image_file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)

        topk_probs, topk_indices = torch.topk(probs, k)

    results = []
    for i in range(k):
        label = le.inverse_transform([topk_indices[0][i].item()])[0]
        confidence = topk_probs[0][i].item() * 100
        results.append((label, confidence))

    return results

# ---------------- UI Logic ----------------
if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.button("🔍 Predict Breed"):
            
            with st.spinner("Analyzing image with AI model... 🤖"):
                results = predict_topk(uploaded_file)

            st.success("Prediction Completed!")

            st.markdown("### 🏆 Top 3 Predictions")

            for i, (breed, conf) in enumerate(results):
                st.markdown(f"""
                <div style="
                    background: rgba(0,0,0,0.4);
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 10px;
                ">
                    <h4>#{i+1} 🐕 {breed}</h4>
                </div>
                """, unsafe_allow_html=True)

                st.progress(int(conf))
                st.write(f"Confidence: {conf:.2f}%")
