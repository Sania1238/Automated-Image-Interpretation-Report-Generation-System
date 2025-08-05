import streamlit as st
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Configure page FIRST - before any other streamlit commands or imports
st.set_page_config(
    page_title="AI Medical Image Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables from .env file
load_dotenv()

# Debugging: Test if GOOGLE_API_KEY is loaded
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    st.success(f"✅ Google API Key loaded: {api_key[:4]}...{api_key[-4:]}")
else:
    st.error("❌ Google API Key not found")

# Add src directory to path AFTER page config
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Now import modules
try:
    from model_utils import load_model, predict_image, get_class_labels
    from report_generator import generate_report
    from image_processor import preprocess_image, display_image_info
    from pdf_utils import create_pdf_report
    from ui_components import create_sidebar, display_results
    MODULES_LOADED = True
except ImportError as e:
    st.error(f"Module import error: {e}")
    MODULES_LOADED = False

def main():
    st.title("🏥 AI Medical Image Analysis System")
    st.markdown("### Automated Chest X-Ray Analysis with AI-Generated Reports")

    if not MODULES_LOADED:
        st.error("❌ Some modules failed to load. Check the error above.")
        st.stop()

    patient_info = create_sidebar()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📄 Upload X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear chest X-ray image for AI analysis"
        )

        if uploaded_file is not None:
            image = display_image_info(uploaded_file)

            if image is not None:
                if st.button("🔍 Analyze Image"):
                    analyze_image(image, patient_info)

    with col2:
        st.header("📊 Analysis Results")
        display_analysis_results()

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p><strong>⚠️ Medical Disclaimer:</strong> This system is for educational purposes only. 
        Always consult qualified medical professionals for diagnosis and treatment.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def analyze_image(image, patient_info):
    with st.spinner("🧐 AI is analyzing the image..."):
        try:
            model = load_model()
            if model is None:
                st.error("❌ Could not load the AI model. Please check if the model file exists.")
                return

            processed_image = preprocess_image(image)
            prediction, confidence, all_predictions = predict_image(model, processed_image)

            st.session_state.update({
                'prediction': prediction,
                'confidence': confidence,
                'all_predictions': all_predictions,
                'image': image,
                'patient_info': patient_info,
                'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            st.success("✅ Analysis completed successfully!")

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")

def display_analysis_results():
    if 'prediction' not in st.session_state:
        st.info("👆 Upload an X-ray image and click 'Analyze Image' to see results")
        return

    display_results(
        st.session_state.prediction,
        st.session_state.confidence,
        st.session_state.all_predictions
    )

    st.subheader("📋 AI-Generated Medical Report")

    with st.spinner("🤖 Generating detailed report..."):
        try:
            report = generate_report(
                st.session_state.prediction,
                st.session_state.confidence,
                st.session_state.patient_info
            )

            st.text_area(
                "Generated Report",
                report,
                height=300,
                help="AI-generated medical report based on image analysis"
            )

            st.session_state.report = report

        except Exception as e:
            st.error(f"❌ Error generating report: {str(e)}")
            return

    st.subheader("📄 Download Report")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 Generate PDF Report"):
            generate_pdf()

    with col2:
        if 'pdf_buffer' in st.session_state:
            st.download_button(
                label="📥 Download PDF",
                data=st.session_state.pdf_buffer,
                file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

def generate_pdf():
    try:
        with st.spinner("📄 Creating PDF report..."):
            pdf_buffer = create_pdf_report(
                st.session_state.image,
                st.session_state.prediction,
                st.session_state.confidence,
                st.session_state.report,
                st.session_state.patient_info,
                st.session_state.analysis_time
            )

            st.session_state.pdf_buffer = pdf_buffer.getvalue()
            st.success("✅ PDF report generated successfully!")

    except Exception as e:
        st.error(f"❌ Error creating PDF: {str(e)}")

if __name__ == "__main__":
    main()
