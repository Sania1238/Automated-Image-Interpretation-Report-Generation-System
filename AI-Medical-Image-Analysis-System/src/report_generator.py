import streamlit as st
import os
from datetime import datetime

def generate_report(prediction, confidence, patient_info=None):
    """
    Generate detailed medical report using Gemini 2.5 Flash
    
    Args:
        prediction: Predicted disease class
        confidence: Confidence score
        patient_info: Optional patient information dictionary
    
    Returns:
        str: Generated medical report
    """
    try:
        # Try Gemini first
        return generate_gemini_report(prediction, confidence, patient_info)
    except Exception as e:
        st.warning(f"⚠️ Gemini API unavailable: {str(e)}")
        st.info("🔄 Falling back to enhanced template-based report...")
        return generate_fallback_report(prediction, confidence, patient_info)

def generate_gemini_report(prediction, confidence, patient_info=None):
    """Generate report using Gemini 2.5 Flash"""
    
    try:
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        # Get API key from secrets or environment
        # For Streamlit deployment, it's best to use st.secrets
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise Exception("Google API key not found. Set it in your environment.")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Build patient context
        patient_context = build_patient_context(patient_info)
        
        # Create comprehensive prompt
        prompt = create_gemini_prompt(prediction, confidence, patient_context)
        
        # --- SOLUTION: ADJUST SAFETY SETTINGS ---
        # This tells the model not to block content for these categories.
        # Use with caution and only in controlled applications.
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # Generate report
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1000,
                top_p=0.8
            ),
            safety_settings=safety_settings  # <-- Add this line
        )
        
        # --- ENHANCED DEBUGGING ---
        # Instead of failing on response.text, check the parts first
        if response.parts:
            return response.text
        else:
            # If there are no parts, the model still refused to answer.
            # Print the prompt feedback to understand why.
            st.error(f"Gemini did not return content. Finish Reason: {response.candidates[0].finish_reason}")
            st.error(f"Prompt Feedback: {response.prompt_feedback}")
            raise Exception("Content generation failed despite safety overrides. Check the prompt or model configuration.")

    except ImportError:
        raise Exception("'google-generativeai' package not installed. Please run 'pip install google-generativeai'")
    except Exception as e:
        # Pass a more informative error message up
        raise Exception(f"An error occurred with the Gemini API: {str(e)}")

def create_gemini_prompt(prediction, confidence, patient_context):
    """Create a re-engineered, safer prompt for Gemini."""
    
    condition_guidance = get_condition_guidance(prediction)
    
    # Re-framed prompt to be a co-pilot, not the primary diagnostician.
    prompt = f"""
    You are an expert assistant for a radiologist, skilled at structuring AI-driven analysis into a professional report format.

    An AI image analysis model has processed a chest X-ray and provided the following preliminary result:
    
    {patient_context}
    
    AI ANALYSIS RESULTS:
    - Predicted Condition: {prediction}
    - AI Confidence Level: {confidence:.1%}
    - Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    Based on the predicted condition '{prediction}', please draft a radiology report using the following guidance and structure. You are to format the information professionally, not to make a diagnosis.

    REPORT STRUCTURE TO FOLLOW:
    
    CHEST X-RAY INTERPRETATION REPORT
    
    CLINICAL INDICATION: Evaluation of chest for potential abnormalities.
    
    TECHNIQUE: Standard chest radiography.
    
    FINDINGS: 
    [Using the guidance below, describe the typical radiological findings for '{prediction}'.]
    Guidance for Findings: {condition_guidance['findings']}
    
    IMPRESSION: 
    [Using the guidance below, write a concise summary for '{prediction}'.]
    Guidance for Impression: {condition_guidance['impression']}
    
    RECOMMENDATIONS: 
    [Using the guidance below, provide a numbered list of appropriate recommendations for '{prediction}'.]
    Guidance for Recommendations: {condition_guidance['recommendations']}
    
    DISCLAIMER: This report was generated with the assistance of an AI model and should be reviewed and validated by a qualified radiologist before being used for clinical decision-making.
    """
    
    return prompt

def get_condition_guidance(prediction):
    """Get condition-specific guidance for report generation"""
    
    guidance = {
        'COVID': {
            'findings': """
            Describe bilateral ground-glass opacities with peripheral and lower lobe distribution.
            Mention the typical appearance of COVID-19 pneumonia.
            Note any associated findings like air bronchograms or consolidation.
            Comment on cardiac silhouette and pleural spaces.
            """,
            'impression': """
            State findings consistent with COVID-19 pneumonia.
            Mention the bilateral peripheral pattern typical of viral pneumonia.
            """,
            'recommendations': """
            Include RT-PCR testing confirmation, isolation protocols,
            clinical correlation with symptoms, follow-up imaging timeline,
            and consideration of chest CT if clinically indicated.
            """
        },
        
        'Viral Pneumonia': {
            'findings': """
            Describe bilateral interstitial or mixed alveolar-interstitial infiltrates.
            Note the diffuse distribution pattern typical of viral etiology.
            Differentiate from bacterial pneumonia appearance.
            Comment on any associated findings.
            """,
            'impression': """
            State findings consistent with viral pneumonia.
            Note the bilateral interstitial pattern.
            """,
            'recommendations': """
            Include supportive care measures, symptom monitoring,
            follow-up imaging schedule, clinical evaluation,
            and consideration of antiviral therapy if specific virus identified.
            """
        },
        
        'Lung_Opacity': {
            'findings': """
            Describe the location, extent, and characteristics of the opacities.
            Consider differential diagnosis including infection, inflammation, or fluid.
            Note any associated findings like air bronchograms or volume loss.
            Comment on distribution pattern.
            """,
            'impression': """
            State presence of lung opacities with differential diagnosis.
            Mention need for clinical correlation.
            """,
            'recommendations': """
            Include clinical correlation with symptoms and vital signs,
            laboratory studies (CBC, inflammatory markers),
            consideration of chest CT for better characterization,
            and appropriate follow-up imaging timeline.
            """
        },
        
        'Normal': {
            'findings': """
            Confirm clear lung fields bilaterally with no consolidation.
            Note normal cardiac silhouette and mediastinal contours.
            Comment on normal diaphragmatic contours and costophrenic angles.
            State no acute abnormalities are present.
            """,
            'impression': """
            State normal chest radiograph with no acute cardiopulmonary abnormalities.
            """,
            'recommendations': """
            Include routine follow-up as clinically appropriate,
            continued clinical monitoring if symptomatic,
            no immediate imaging follow-up required,
            and age-appropriate screening recommendations.
            """
        }
    }
    
    return guidance.get(prediction, guidance['Normal'])

def build_patient_context(patient_info):
    """Build patient context string from provided information"""
    if not patient_info or not any(patient_info.values()):
        return "PATIENT INFORMATION: Not provided"
    
    context_parts = ["PATIENT INFORMATION:"]
    for key, value in patient_info.items():
        if value:
            context_parts.append(f"- {key}: {value}")
    
    return "\n".join(context_parts)

def generate_fallback_report(prediction, confidence, patient_info=None):
    """Generate enhanced template-based report when Gemini is unavailable"""
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    patient_section = build_patient_context(patient_info)
    
    reports = {
        'COVID': f"""CHEST X-RAY INTERPRETATION REPORT

{patient_section}

CLINICAL INDICATION: Evaluation for suspected COVID-19 pneumonia

TECHNIQUE: Standard chest radiography

FINDINGS: The chest radiograph demonstrates findings consistent with COVID-19 pneumonia (AI confidence: {confidence:.1%}). Bilateral ground-glass opacities are observed, predominantly in the peripheral and lower lobe distribution. The pattern is characteristic of viral pneumonia with COVID-19 features. The cardiac silhouette appears normal in size and contour. No pleural effusion or pneumothorax is identified. The mediastinal contours are unremarkable.

IMPRESSION: Radiographic findings highly suggestive of COVID-19 pneumonia with bilateral peripheral ground-glass opacities.

RECOMMENDATIONS:
1. RT-PCR testing for COVID-19 confirmation and clinical correlation
2. Patient isolation per institutional COVID-19 protocols
3. Follow-up chest imaging in 7-10 days or if clinical condition deteriorates
4. Consider chest CT for better characterization if symptoms worsen
5. Monitor oxygen saturation and respiratory status closely

Report generated: {current_time}""",

        'Viral Pneumonia': f"""CHEST X-RAY INTERPRETATION REPORT

{patient_section}

CLINICAL INDICATION: Evaluation for suspected viral pneumonia

TECHNIQUE: Standard chest radiography

FINDINGS: The chest radiograph shows findings consistent with viral pneumonia (AI confidence: {confidence:.1%}). Bilateral interstitial infiltrates are observed with a diffuse pattern throughout both lung fields. The appearance suggests viral etiology rather than bacterial pneumonia. The cardiac silhouette is within normal limits. No significant pleural effusion is noted.

IMPRESSION: Findings consistent with viral pneumonia, characterized by bilateral interstitial infiltrates.

RECOMMENDATIONS:
1. Clinical correlation with symptoms and vital signs
2. Supportive care and symptomatic treatment as indicated
3. Follow-up chest radiograph in 7-10 days to assess progression
4. Consider viral studies if specific pathogen identification needed
5. Monitor for complications and respiratory deterioration

Report generated: {current_time}""",

        'Lung_Opacity': f"""CHEST X-RAY INTERPRETATION REPORT

{patient_section}

CLINICAL INDICATION: Evaluation of lung opacities

TECHNIQUE: Standard chest radiography

FINDINGS: The chest radiograph reveals lung opacities (AI confidence: {confidence:.1%}). Areas of increased density are noted, suggesting possible infectious process, inflammatory changes, or fluid accumulation. The distribution and characteristics require clinical correlation for definitive diagnosis. The cardiac silhouette appears normal. Costophrenic angles are preserved.

IMPRESSION: Lung opacities present with differential diagnosis including pneumonia, pulmonary edema, or inflammatory process.

RECOMMENDATIONS:
1. Clinical correlation with patient symptoms, vital signs, and physical examination
2. Complete blood count and inflammatory markers (CRP, ESR, procalcitonin)
3. Consider chest CT for better characterization of opacities
4. Follow-up imaging in 48-72 hours to assess response to treatment
5. Appropriate antimicrobial therapy if infectious etiology suspected

Report generated: {current_time}""",

        'Normal': f"""CHEST X-RAY INTERPRETATION REPORT

{patient_section}

CLINICAL INDICATION: Routine chest evaluation

TECHNIQUE: Standard chest radiography

FINDINGS: The chest radiograph appears normal (AI confidence: {confidence:.1%}). The lungs are clear bilaterally with no evidence of consolidation, pneumothorax, or pleural effusion. The cardiac silhouette is normal in size and configuration. The mediastinal contours are unremarkable. The diaphragmatic contours are normal and the costophrenic angles are sharp.

IMPRESSION: Normal chest radiograph. No acute cardiopulmonary abnormalities detected.

RECOMMENDATIONS:
1. No immediate follow-up imaging required unless clinically indicated
2. Continue routine health maintenance and age-appropriate screening
3. Return for imaging if respiratory symptoms develop
4. Clinical follow-up as deemed appropriate by treating physician

Report generated: {current_time}"""
    }
    
    return reports.get(prediction, f"Report generation error for condition: {prediction}")

def test_gemini_connection():
    """Test Gemini API connection"""
    try:
        import google.generativeai as genai
        
        api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return False, "API key not found"
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        response = model.generate_content("Test connection: respond with 'Connected'")
        return True, response.text
        
    except Exception as e:
        return False, str(e)