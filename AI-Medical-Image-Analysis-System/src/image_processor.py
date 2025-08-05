import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

def preprocess_image(image):
    """
    Preprocess image for model prediction
    
    Args:
        image: PIL Image object
    
    Returns:
        numpy array: Preprocessed image ready for model
    """
    try:
        # Resize image to model input size (224x224 for MobileNetV2)
        target_size = (224, 224)
        image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if image_resized.mode != 'RGB':
            image_resized = image_resized.convert('RGB')
        
        # Convert to numpy array
        img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize pixel values to [0, 1] range
        img_array = img_array / 255.0
        
        return img_array
        
    except Exception as e:
        raise Exception(f"Image preprocessing failed: {str(e)}")

def display_image_info(uploaded_file):
    """
    Display uploaded image with information
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        PIL Image: Opened image object
    """
    try:
        # Open image
        image = Image.open(uploaded_file)
        
        # Display image
        st.image(
            image, 
            caption=f"Uploaded: {uploaded_file.name}",
            use_container_width=True
        )
        
        # Display image information
        with st.expander("📊 Image Information"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
                st.write(f"**Mode:** {image.mode}")
            
            with col2:
                st.write(f"**Format:** {image.format}")
                st.write(f"**File Size:** {uploaded_file.size / 1024:.1f} KB")
                st.write(f"**Aspect Ratio:** {image.size[0]/image.size[1]:.2f}")
        
        # Validate image
        validation_result = validate_image(image)
        if not validation_result['is_valid']:
            st.warning(f"⚠️ {validation_result['message']}")
        else:
            st.success(f"✅ {validation_result['message']}")
        
        return image
        
    except Exception as e:
        st.error(f"❌ Error loading image: {str(e)}")
        return None

def validate_image(image):
    """
    Validate if image is suitable for medical analysis
    
    Args:
        image: PIL Image object
    
    Returns:
        dict: Validation result with is_valid and message
    """
    try:
        width, height = image.size
        
        # Check minimum resolution
        if width < 100 or height < 100:
            return {
                'is_valid': False,
                'message': 'Image resolution too low. Please upload a higher quality image.'
            }
        
        # Check if image is too small
        if width < 224 and height < 224:
            return {
                'is_valid': True,
                'message': 'Image will be upscaled for analysis. Consider using higher resolution images for better results.'
            }
        
        # Check aspect ratio
        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return {
                'is_valid': True,
                'message': 'Unusual aspect ratio detected. Ensure the image shows a complete chest X-ray.'
            }
        
        # Check if image is grayscale (common for X-rays)
        if image.mode == 'L':
            return {
                'is_valid': True,
                'message': 'Grayscale X-ray image detected - suitable for analysis.'
            }
        
        # Check if image is RGB
        if image.mode == 'RGB':
            return {
                'is_valid': True,
                'message': 'Color image detected - will be processed for analysis.'
            }
        
        return {
            'is_valid': True,
            'message': 'Image appears suitable for medical analysis.'
        }
        
    except Exception as e:
        return {
            'is_valid': False,
            'message': f'Image validation error: {str(e)}'
        }