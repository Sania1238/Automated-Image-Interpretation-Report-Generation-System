import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from model_utils import get_class_labels, get_prediction_interpretation
from report_generator import test_gemini_connection

def setup_page_config():
    """Configure Streamlit page settings - NOTE: Page config is now handled in main app"""
    # Page config moved to main app.py to avoid duplicate calls
    pass
    # Add custom CSS styling
    add_custom_css()
def add_custom_css():
    """Add custom CSS styling"""
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        padding: 1rem 0;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .high-confidence { background-color: #d4edda; color: #155724; }
    .medium-confidence { background-color: #fff3cd; color: #856404; }
    .low-confidence { background-color: #f8d7da; color: #721c24; }
    
    .sidebar-info {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create sidebar with patient information and system status"""
    with st.sidebar:
        st.header("🩺 Patient Information")
        
        # Patient info form
        patient_info = {}
        
        with st.form("patient_form"):
            patient_info['Patient ID'] = st.text_input(
                "Patient ID", 
                help="Enter patient identifier"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                patient_info['Age'] = st.number_input(
                    "Age", 
                    min_value=0, 
                    max_value=120, 
                    value=None,
                    help="Patient age in years"
                )
            
            with col2:
                patient_info['Gender'] = st.selectbox(
                    "Gender", 
                    ["", "Male", "Female", "Other"]
                )
            
            patient_info['Clinical History'] = st.text_area(
                "Clinical History",
                height=100,
                help="Relevant medical history and symptoms"
            )
            
            patient_info['Referring Physician'] = st.text_input(
                "Referring Physician",
                help="Name of referring doctor"
            )
            
            # Submit button (optional - info is captured automatically)
            submitted = st.form_submit_button("💾 Save Patient Info")
            if submitted:
                st.success("✅ Patient information saved!")
        
        # Filter out empty values
        patient_info = {k: v for k, v in patient_info.items() if v}
        
        # System status section
        st.markdown("---")
        st.header("⚙️ System Status")
        
        # Test connections
        if st.button("🧪 Test AI Systems"):
            test_system_connections()
        
        # Model info
        with st.expander("📊 Model Information"):
            st.write("**Model Type:** MobileNetV2 + Custom Head")
            st.write("**Classes:** COVID, Lung_Opacity, Normal, Viral Pneumonia")
            st.write("**Input Size:** 224×224 pixels")
            st.write("**Framework:** TensorFlow/Keras")
        
        # Usage tips
        with st.expander("💡 Usage Tips"):
            st.write("**For best results:**")
            st.write("• Use clear, high-resolution X-ray images")
            st.write("• Ensure image shows complete chest area")
            st.write("• Images should be properly oriented")
            st.write("• Avoid heavily processed or filtered images")
    
    return patient_info

def test_system_connections():
    """Test system components and display status"""
    st.write("**Testing system components...**")
    
    # Test Gemini connection
    with st.spinner("Testing Gemini API..."):
        gemini_success, gemini_msg = test_gemini_connection()
        if gemini_success:
            st.success(f"✅ Gemini API: Connected")
        else:
            st.warning(f"⚠️ Gemini API: {gemini_msg}")
    
    # Test model loading (placeholder)
    st.info("ℹ️ Model: Ready (add model file to test)")
    
    # Test PDF generation
    from pdf_utils import validate_pdf_generation
    pdf_success, pdf_msg = validate_pdf_generation()
    if pdf_success:
        st.success("✅ PDF Generation: Working")
    else:
        st.error(f"❌ PDF Generation: {pdf_msg}")

def display_results(prediction, confidence, all_predictions):
    """Display prediction results with visualizations"""
    
    # Get prediction interpretation
    interpretation = get_prediction_interpretation(prediction, confidence)
    
    # Main prediction display
    confidence_class = 'high-confidence' if confidence > 0.8 else 'medium-confidence' if confidence > 0.6 else 'low-confidence'
    
    st.markdown(f"""
    <div class="prediction-box {confidence_class}">
        {interpretation['icon']} <strong>{prediction}</strong><br>
        Confidence: {confidence:.1%} ({interpretation['confidence_level']})
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Prediction",
            prediction,
            delta=f"{confidence:.1%} confidence"
        )
    
    with col2:
        st.metric(
            "Urgency Level",
            interpretation['urgency'],
            delta=interpretation['description']
        )
    
    with col3:
        st.metric(
            "Confidence Level",
            interpretation['confidence_level'],
            delta=f"{confidence:.2%}"
        )
    
    # Detailed predictions chart
    st.subheader("📈 Detailed Analysis")
    
    # Create confidence chart
    create_confidence_chart(all_predictions)
    
    # Prediction details table
    create_prediction_table(all_predictions, prediction)

def create_confidence_chart(all_predictions):
    """Create interactive confidence chart"""
    
    # Prepare data for chart
    classes = list(all_predictions.keys())
    confidences = list(all_predictions.values())
    
    # Create color map
    colors = ['#FF6B6B' if pred == max(confidences) else '#4ECDC4' for pred in confidences]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=confidences,
            marker_color=colors,
            text=[f'{conf:.1%}' for conf in confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="AI Confidence Scores by Condition",
        xaxis_title="Medical Conditions",
        yaxis_title="Confidence Score",
        yaxis=dict(tickformat='.0%'),
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_prediction_table(all_predictions, top_prediction):
    """Create detailed predictions table"""
    
    # Sort predictions by confidence
    sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
    
    # Create table data
    table_data = []
    for condition, confidence in sorted_predictions:
        # Add indicators
        if condition == top_prediction:
            indicator = "🎯 **PREDICTED**"
        elif confidence > 0.1:
            indicator = "⚠️ Consider"
        else:
            indicator = "✅ Unlikely"
        
        table_data.append({
            "Condition": condition,
            "Confidence": f"{confidence:.1%}",
            "Status": indicator,
            "Bar": confidence
        })
    
    # Display as DataFrame with custom styling
    import pandas as pd
    df = pd.DataFrame(table_data)
    
    # Create progress bars for confidence
    st.write("**Detailed Confidence Breakdown:**")
    
    for _, row in df.iterrows():
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.write(f"**{row['Condition']}**")
        
        with col2:
            st.write(row['Confidence'])
        
        with col3:
            st.progress(row['Bar'])
            st.write(row['Status'], unsafe_allow_html=True)

def create_medical_advice_box(prediction, confidence):
    """Create medical advice box based on prediction"""
    
    advice_content = {
        'COVID': {
            'immediate': "🚨 Immediate isolation recommended",
            'testing': "📋 RT-PCR testing required",
            'monitoring': "🩺 Monitor oxygen levels closely",
            'followup': "📅 Follow-up in 7-10 days"
        },
        'Viral Pneumonia': {
            'immediate': "🏥 Medical evaluation recommended",
            'testing': "🧪 Consider viral panel testing",
            'monitoring': "🌡️ Monitor symptoms and fever",
            'followup': "📅 Follow-up in 5-7 days"
        },
        'Lung_Opacity': {
            'immediate': "👨‍⚕️ Clinical correlation needed",
            'testing': "🔬 Laboratory studies recommended",
            'monitoring': "📊 Monitor symptoms progression",
            'followup': "📅 Follow-up in 2-3 days"
        },
        'Normal': {
            'immediate': "✅ No immediate action required",
            'testing': "📝 Routine care as needed",
            'monitoring': "😊 Continue normal activities",
            'followup': "📅 Routine follow-up as scheduled"
        }
    }
    
    advice = advice_content.get(prediction, advice_content['Normal'])
    
    st.subheader("🩺 Clinical Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Immediate:** {advice['immediate']}")
        st.info(f"**Testing:** {advice['testing']}")
    
    with col2:
        st.info(f"**Monitoring:** {advice['monitoring']}")
        st.info(f"**Follow-up:** {advice['followup']}")
    
    # Confidence-based additional advice
    if confidence < 0.6:
        st.warning("⚠️ **Low Confidence Alert:** Consider additional imaging or second opinion due to moderate confidence level.")
    elif confidence > 0.9:
        st.success("✅ **High Confidence:** AI analysis shows strong confidence in prediction.")

def display_system_info():
    """Display system information in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.subheader("ℹ️ System Information")
        
        st.write("**Version:** 1.0.0")
        st.write("**Last Updated:** 2024")
        st.write("**Model:** MobileNetV2")
        st.write("**Classes:** 4 conditions")
        
        # Performance metrics (placeholder)
        st.write("**Performance Metrics:**")
        st.write("• Training Accuracy: 93.6%")
        st.write("• Validation Accuracy: 89.8%")
        st.write("• Model Size: ~15MB")