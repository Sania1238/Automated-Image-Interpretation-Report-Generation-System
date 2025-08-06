# 🏥 AI Medical Image Analysis System

A comprehensive medical imaging analysis system that uses CNN for chest X-ray classification and LLM for automated report generation.

## 📋 Project Overview

This system combines computer vision and natural language processing to:
- Analyze chest X-ray images using a trained CNN model
- Classify images into 4 categories: COVID-19, Viral Pneumonia, Lung Opacity, Normal
- Generate detailed medical reports using Gemini 2.5 Flash LLM
- Create downloadable PDF reports with analysis results

## 🏗️ Project Structure

```
medical_imaging_project/
├── app.py                 # Main Streamlit application
├── src/
│   ├── __init__.py
│   ├── model_utils.py     # CNN model loading & prediction
│   ├── report_generator.py # LLM integration for reports
│   ├── image_processor.py # Image preprocessing utilities
│   ├── pdf_utils.py       # PDF report generation
│   └── ui_components.py   # Streamlit UI components
├── models/
│   └── medical_model.h5   # Your trained CNN model (add this file)
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .env                  # Environment variables (create this)
```

## 🚀 Quick Start

### 1.Setup

# Create the virtual environment
python - m venv env
## Activate the environment
env\Scripts\activate
## Install the dependencies
pip install -r requirements.txt
```

### 2. Add Your Model
- Place your trained `medical_model.h5` file in the `models/` directory
- The model should accept 224×224×3 input images
- Output should be 4 classes: [COVID, Lung_Opacity, Normal, Viral Pneumonia]

### 3. Setup Environment Variables
Create a `.env` file in the root directory:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Run the Application
```bash
streamlit run app.py
```

## 🔧 Configuration

### Getting Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in and create a new API key
3. Add it to your `.env` file or Streamlit secrets

### For Streamlit Cloud Deployment
Add to your Streamlit secrets:
```toml
GOOGLE_API_KEY = "your_api_key_
