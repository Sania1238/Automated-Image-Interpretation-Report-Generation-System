import tensorflow as tf
import sys
import os

print("🔍 TensorFlow Version Check")
print("=" * 40)
print(f"TensorFlow version: {tf.__version__}")
print(f"Python version: {sys.version}")
print(f"Model path exists: {os.path.exists('models/medical_model.h5')}")

# Check if model file exists and get its info
model_path = 'models/medical_model.h5'
if os.path.exists(model_path):
    print(f"Model file size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    
    # Try to load with different methods
    print("\n🧪 Testing model loading methods...")
    
    # Method 1: Standard loading
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Standard loading: SUCCESS")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Total params: {model.count_params():,}")
    except Exception as e:
        print(f"❌ Standard loading failed: {e}")
    
    # Method 2: Loading with compile=False
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Loading without compile: SUCCESS")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Loading without compile failed: {e}")
    
    # Method 3: Loading with custom objects
    try:
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=None,
            compile=False,
            safe_mode=False
        )
        print("✅ Loading with safe_mode=False: SUCCESS")
    except Exception as e:
        print(f"❌ Loading with safe_mode=False failed: {e}")

else:
    print("❌ Model file not found!")
    print("Available files in models directory:")
    models_dir = 'models'
    if os.path.exists(models_dir):
        files = os.listdir(models_dir)
        for file in files:
            print(f"   - {file}")
    else:
        print("   models directory doesn't exist")

print("\n💡 Recommendations:")
if tf.__version__.startswith('2.13'):
    print("- You have TensorFlow 2.13 - try upgrading to latest version")
    print("- Run: pip install --upgrade tensorflow")
elif tf.__version__.startswith('2.15') or tf.__version__.startswith('2.16'):
    print("- You have a recent TensorFlow version")
    print("- The model might be from an older version")
else:
    print(f"- TensorFlow {tf.__version__} detected")
    print("- Consider upgrading to TensorFlow 2.15+")

print("\n🔧 Quick fixes to try:")
print("1. pip install --upgrade tensorflow")
print("2. Re-save your model with current TensorFlow version")
print("3. Use compile=False when loading model")