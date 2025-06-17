import streamlit as st
from PIL import Image
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from inference_sdk import InferenceHTTPClient
import google.generativeai as genai
from config import GEMINI_API_KEY
from gemini_prompt import generate_response
from fpdf import FPDF

# Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="2gl34PLLNsu76bEcsRCy"
)

# Gemini Configuration
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Function to create PDF summary
def create_pdf(disease, advice):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Arial", 'B', 14)
    pdf.multi_cell(0, 10, "Chest X-ray Disease Detection Report", align='C')
    pdf.ln()

    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 10, f"Detected Disease: {disease}", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, f"\nAI-Generated Medical Advice:\n\n{advice}")

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_file.name)
    return tmp_file.name

# Streamlit App
st.set_page_config(page_title="🩻 Chest X-ray Analyzer", layout="centered")
st.title("🩺 Chest X-ray Disease Detection & Cure Assistant")
st.write("Upload a chest X-ray to detect **Covid, Pneumonia, Tuberculosis, or Normal** and receive AI-driven advice.")

uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded X-ray", use_container_width=True)

    # Save uploaded file to a temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    uploaded_file.seek(0)  # ✅ Reset stream to beginning
    temp_file.write(uploaded_file.read())
    temp_file.flush()

    with st.spinner("🔍 Detecting disease using Roboflow YOLO..."):
        try:
            result = CLIENT.infer(temp_file.name, model_id="lung-disease-detection/1")
            predictions = result.get("predictions", [])

            if predictions:
                top_prediction = max(predictions, key=lambda x: x['confidence'])
                disease = top_prediction['class']
                st.success(f"🎯 Detected: **{disease}**")
            else:
                disease = "Normal"
                st.success("✅ Detected: **Normal**")
        except Exception as e:
            st.error(f"❌ Error during inference: {str(e)}")
            st.stop()

    # Bounding Box Drawing
    img = Image.open(temp_file.name)
    fig, ax = plt.subplots()
    ax.imshow(img)

    for pred in predictions:
        if pred['confidence'] >= 0.5:
            x, y = pred['x'], pred['y']
            w, h = pred['width'], pred['height']
            label = pred['class']

            rect = patches.Rectangle(
                (x - w / 2, y - h / 2), w, h,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)

            ax.text(
                x - w / 2, y - h / 2 - 10,
                f"{label} ({pred['confidence']:.2f})",
                color='lime', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5)
            )

    ax.axis("off")
    st.pyplot(fig)

    with st.spinner("💡 Generating medical advice using Gemini AI..."):
        advice = generate_response(model, disease)
        st.markdown("### 💬 AI Recommendation")
        st.markdown(advice)

    with st.spinner("📄 Creating PDF Summary..."):
        pdf_path = create_pdf(disease, advice)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download PDF Summary",
                data=f,
                file_name=f"{disease}_diagnosis_summary.pdf",
                mime="application/pdf"
            )
