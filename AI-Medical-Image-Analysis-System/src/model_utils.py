import tensorflow as tf
import numpy as np
import streamlit as st
import os

# Class labels from your COVID-19 dataset
CLASS_LABELS = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

# Model file path
MODEL_PATH = 'models/medical_model.h5'

def load_model():
    """
    Load the trained CNN model
    Returns the loaded model or None if loading fails
    """
    try:
        if not os.path.exists(MODEL_PATH):
            st.warning(f"⚠️ Model file not found at {MODEL_PATH}")
            st.info("Please upload your trained model file to the 'models/' directory")
            return None
        
        # Load the model
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Verify model architecture
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        st.success(f"✅ Model loaded successfully!")
        st.info(f"📊 Model Info: Input shape: {input_shape}, Output classes: {output_shape[1]}")
        
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def predict_image(model, processed_image):
    """
    Make prediction on preprocessed image
    
    Args:
        model: Loaded Keras model
        processed_image: Preprocessed image array
    
    Returns:
        tuple: (predicted_class, confidence, all_predictions)
    """
    try:
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get the predicted class
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_LABELS[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        # Create dictionary of all predictions for display
        all_predictions = {
            CLASS_LABELS[i]: float(predictions[0][i]) 
            for i in range(len(CLASS_LABELS))
        }
        
        return predicted_class, float(confidence), all_predictions
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

def get_class_labels():
    """Return the class labels"""
    return CLASS_LABELS

def get_model_info():
    """
    Get information about the loaded model
    Returns dictionary with model details
    """
    try:
        model = load_model()
        if model is None:
            return None
        
        return {
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'total_params': model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
            'layers': len(model.layers)
        }
        
    except Exception as e:
        st.error(f"Error getting model info: {str(e)}")
        return None

def validate_prediction_confidence(confidence, threshold=0.5):
    """
    Validate if prediction confidence meets minimum threshold
    
    Args:
        confidence: Prediction confidence score
        threshold: Minimum confidence threshold
    
    Returns:
        tuple: (is_valid, message)
    """
    if confidence >= threshold:
        return True, "High confidence prediction"
    elif confidence >= 0.3:
        return True, "Moderate confidence - consider additional testing"
    else:
        return False, "Low confidence - manual review recommended"

def get_prediction_interpretation(predicted_class, confidence):
    """
    Get interpretation of the prediction results
    
    Args:
        predicted_class: Predicted disease class
        confidence: Confidence score
    
    Returns:
        dict: Interpretation details
    """
    interpretations = {
        'COVID': {
            'description': 'COVID-19 pneumonia detected',
            'urgency': 'High',
            'color': 'red',
            'icon': '🦠'
        },
        'Viral Pneumonia': {
            'description': 'Viral pneumonia detected',
            'urgency': 'High',
            'color': 'orange',
            'icon': '🫁'
        },
        'Lung_Opacity': {
            'description': 'Lung opacities detected',
            'urgency': 'Medium',
            'color': 'yellow',
            'icon': '⚠️'
        },
        'Normal': {
            'description': 'No abnormalities detected',
            'urgency': 'Low',
            'color': 'green',
            'icon': '✅'
        }
    }
    
    base_info = interpretations.get(predicted_class, {
        'description': 'Unknown condition',
        'urgency': 'Unknown',
        'color': 'gray',
        'icon': '❓'
    })
    
    base_info['confidence'] = confidence
    base_info['confidence_level'] = 'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low'
    
    return base_info