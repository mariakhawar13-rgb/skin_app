# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import cv2
import sys

# ========== PAGE SETUP ==========
st.set_page_config(
    page_title="Skin Cancer Classifier",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Skin Cancer Classification")
st.markdown("Upload an image of a skin lesion for AI analysis")

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    """Load model with fallback options"""
    try:
        # Method 1: Try .keras format (Keras 3)
        model = tf.keras.models.load_model('skin_cancer_model.keras')
        st.success("✅ Model loaded (.keras format)")
        return model
    except Exception as e:
        st.warning(f"⚠️ .keras loading failed: {str(e)[:100]}")
        
        try:
            # Method 2: Try loading from architecture + weights
            with open('model_architecture.json', 'r') as f:
                model_json = f.read()
            model = tf.keras.models.model_from_json(model_json)
            model.load_weights('model_weights.weights.h5')
            st.success("✅ Model loaded (architecture + weights)")
            return model
        except Exception as e2:
            st.error(f"❌ All loading methods failed")
            st.info("Running in demo mode")
            return None

# ========== LOAD CLASSES ==========
@st.cache_data
def load_classes():
    """Load class names"""
    try:
        with open('class_names.json', 'r') as f:
            return json.load(f)
    except:
        # Fallback to ISIC classes
        return [
            'basal cell carcinoma',
            'seborrheic keratosis', 
            'dermatofibroma',
            'melanoma',
            'nevus',
            'vascular lesion',
            'pigmented benign keratosis',
            'actinic keratosis',
            'squamous cell carcinoma'
        ]

# ========== PREPROCESSING ==========
def preprocess_image(image):
    """Preprocess image for model"""
    try:
        # Convert PIL to numpy
        img = np.array(image)
        
        # Convert to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Resize to 224x224
        img = cv2.resize(img, (224, 224))
        
        # Normalize (simple 0-1 normalization for compatibility)
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        st.error(f"Image processing error: {e}")
        return None

# ========== MAIN APP ==========
def main():
    # Load resources
    model = load_model()
    classes = load_classes()
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # Upload options
        upload_method = st.radio(
            "Select input method:",
            ["Upload File", "Use Camera"],
            horizontal=True
        )
        
        image = None
        
        if upload_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a skin lesion image",
                type=['jpg', 'jpeg', 'png'],
                help="For best results, use clear, well-lit images"
            )
            if uploaded_file:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                except:
                    st.error("Could not open image file")
        
        else:  # Camera
            camera_image = st.camera_input("Take a picture of the skin lesion")
            if camera_image:
                image = Image.open(camera_image)
                st.image(image, caption="Captured Image", use_column_width=True)
    
    with col2:
        st.header("📊 Analysis Results")
        
        if image:
            if model is not None:
                with st.spinner("🔍 Analyzing image..."):
                    # Preprocess and predict
                    processed = preprocess_image(image)
                    
                    if processed is not None:
                        try:
                            predictions = model.predict(processed, verbose=0)[0]
                            
                            # Get top prediction
                            top_idx = np.argmax(predictions)
                            top_class = classes[top_idx]
                            confidence = predictions[top_idx]
                            
                            # Display results
                            st.subheader(f"Primary Diagnosis: **{top_class}**")
                            
                            # Confidence display
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.progress(float(confidence))
                            with col_b:
                                st.metric("Confidence", f"{confidence:.2%}")
                            
                            # Show all predictions
                            with st.expander("📋 View Detailed Predictions", expanded=False):
                                for i, (cls, prob) in enumerate(zip(classes, predictions)):
                                    st.write(f"{i+1}. **{cls}**: {prob:.2%}")
                            
                            # Risk assessment
                            st.subheader("⚠️ Risk Assessment")
                            
                            high_risk_keywords = ['melanoma', 'carcinoma', 'squamous']
                            predicted_lower = top_class.lower()
                            
                            if any(keyword in predicted_lower for keyword in high_risk_keywords):
                                if confidence > 0.7:
                                    st.error("""
                                    **🔴 HIGH RISK INDICATED**
                                    
                                    This prediction suggests a potentially malignant skin lesion.
                                    **Consult a dermatologist immediately.**
                                    """)
                                else:
                                    st.warning("""
                                    **🟡 POTENTIAL CONCERN**
                                    
                                    This lesion shows characteristics that may require medical attention.
                                    **Consider consulting a healthcare professional.**
                                    """)
                            else:
                                st.success("""
                                **🟢 LIKELY BENIGN**
                                
                                This appears to be a non-cancerous or low-risk skin lesion.
                                **Always monitor for changes and consult a doctor if concerned.**
                                """)
                        
                        except Exception as e:
                            st.error(f"Prediction error: {str(e)[:200]}")
                            st.info("Showing class information only")
                            st.write("**Available classes:**")
                            for cls in classes:
                                st.write(f"• {cls}")
            
            else:
                # Demo mode (model not loaded)
                st.info("🎯 **Demo Mode**")
                st.write("Model not loaded. Showing available skin condition classes:")
                for i, cls in enumerate(classes):
                    st.write(f"{i+1}. {cls}")
                
                st.divider()
                st.warning("""
                **If model was loaded, you would see:**
                - AI predictions with confidence scores
                - Risk assessment
                - Detailed analysis
                """)
        
        else:
            st.info("👆 Upload an image or use the camera to get started")
            st.markdown("""
            **Tips for best results:**
            - Use clear, well-lit images
            - Center the lesion in the frame
            - Avoid shadows or glare
            - Include some normal skin around the lesion
            """)
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About This Tool")
        
        st.markdown("""
        ### Model Information
        - **Architecture**: Fine-tuned EfficientNetB0
        - **Framework**: TensorFlow 2.19.0 / Keras 3
        - **Input**: 224×224 RGB images
        - **Output**: 9 skin condition classes
        - **Accuracy**: ~66% (educational/demo use)
        
        ### How It Works
        1. Upload a clear image of a skin lesion
        2. AI analyzes visual patterns
        3. Get instant classification results
        4. Review risk assessment
        
        ### Limitations
        - Accuracy: ~66% (for educational use)
        - Not FDA-approved for medical diagnosis
        - Works best with clear, well-lit images
        """)
        
        st.divider()
        
        st.warning("""
        ### ⚠️ IMPORTANT DISCLAIMER
        
        **This tool is for EDUCATIONAL and DEMONSTRATION purposes only.**
        
        - NOT a substitute for professional medical advice
        - NOT a diagnostic tool
        - Accuracy is limited (~66%)
        - Always consult a qualified healthcare provider
        - Do not make medical decisions based on this tool alone
        
        The developers are not responsible for any medical decisions
        made based on this tool's output.
        """)
        
        st.divider()
        
        # Technical info
        if st.checkbox("Show technical details"):
            st.write("**Loaded files:**")
            loaded_files = []
            for file in ['skin_cancer_model.keras', 'class_names.json', 'requirements.txt']:
                try:
                    if file.endswith('.json'):
                        open(file, 'r').close()
                    else:
                        open(file, 'rb').close()
                    loaded_files.append(f"✅ {file}")
                except:
                    loaded_files.append(f"❌ {file}")
            
            for status in loaded_files:
                st.write(status)

# ========== RUN APP ==========
if __name__ == "__main__":
    main()