import streamlit as st
import os
from PIL import Image
import numpy as np

# Suppress TensorFlow oneDNN logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the trained model
@st.cache_resource
def load_prediction_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'mri_model_trained.keras')
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    return load_model(model_path)

model = load_prediction_model()
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Treatment recommendations with detailed steps
treatments = {
    'glioma': {
        'title': 'Glioma Treatment Plan',
        'overview': 'Gliomas are tumors that arise from glial cells in the brain. Treatment depends on the grade, location, and type of glioma.',
        'steps': [
            '**1. Initial Assessment and Diagnosis**',
            '- Comprehensive neurological examination',
            '- Advanced imaging (MRI, CT scans) for tumor characterization',
            '- Biopsy to determine tumor grade and molecular markers',
            '- Consultation with multidisciplinary team (neurosurgeon, oncologist, neurologist)',

            '**2. Surgical Intervention**',
            '- Maximal safe resection to remove as much tumor as possible',
            '- Awake craniotomy for tumors near eloquent brain areas',
            '- Intraoperative MRI guidance for precise tumor removal',
            '- Postoperative monitoring in ICU',

            '**3. Radiation Therapy**',
            '- External beam radiation therapy (typically 6 weeks)',
            '- Stereotactic radiosurgery for small residual tumors',
            '- Proton therapy for tumors near critical structures',
            '- Fractionated stereotactic radiotherapy',

            '**4. Chemotherapy**',
            '- Temozolomide (TMZ) regimen during and after radiation',
            '- PCV chemotherapy (procarbazine, lomustine, vincristine) for anaplastic gliomas',
            '- Targeted therapies based on molecular profiling (IDH inhibitors, etc.)',

            '**5. Follow-up and Monitoring**',
            '- Regular MRI scans every 3-6 months',
            '- Neurological assessments and quality of life evaluations',
            '- Rehabilitation therapy (physical, occupational, speech)',
            '- Supportive care for side effects and symptoms'
        ],
        'duration': 'Treatment typically spans 6-12 months initially, with lifelong monitoring',
        'success_rate': '5-year survival rates vary by grade: 90%+ for low-grade, 10-30% for high-grade'
    },
    'meningioma': {
        'title': 'Meningioma Treatment Plan',
        'overview': 'Meningiomas are typically benign tumors arising from the meninges. Treatment focuses on complete removal when possible.',
        'steps': [
            '**1. Initial Evaluation**',
            '- Detailed neurological examination',
            '- High-resolution MRI with contrast for tumor characterization',
            '- CT scans to assess bone involvement',
            '- Angiography to evaluate blood supply to the tumor',

            '**2. Surgical Treatment**',
            '- Complete surgical resection (Simpson Grade I-II)',
            '- Craniotomy approach based on tumor location',
            '- Microsurgical techniques for tumor dissection',
            '- Intraoperative neurophysiological monitoring',

            '**3. Radiation Therapy Options**',
            '- Stereotactic radiosurgery (Gamma Knife, CyberKnife) for residual tumors',
            '- Fractionated stereotactic radiotherapy',
            '- Conventional external beam radiation for atypical/malignant meningiomas',
            '- Proton beam therapy for skull base tumors',

            '**4. Medical Management**',
            '- Hormone therapy for hormone-sensitive tumors',
            '- Anti-seizure medications if seizures are present',
            '- Pain management and symptom control',
            '- Management of peritumoral edema',

            '**5. Long-term Follow-up**',
            '- Annual MRI surveillance for 5 years, then every 2-3 years',
            '- Monitoring for tumor recurrence or progression',
            '- Rehabilitation services as needed',
            '- Regular endocrinological evaluation if pituitary function affected'
        ],
        'duration': 'Recovery from surgery: 4-8 weeks, with long-term monitoring',
        'success_rate': '95%+ for benign meningiomas with complete resection'
    },
    'pituitary': {
        'title': 'Pituitary Tumor Treatment Plan',
        'overview': 'Pituitary tumors can affect hormone production and cause various endocrine symptoms. Treatment aims to restore normal pituitary function.',
        'steps': [
            '**1. Comprehensive Evaluation**',
            '- Detailed hormonal assessment (pituitary function tests)',
            '- Visual field testing for tumors affecting optic nerves',
            '- High-resolution MRI with dedicated pituitary protocol',
            '- Consultation with endocrinologist and neurosurgeon',

            '**2. Surgical Treatment**',
            '- Transsphenoidal surgery (preferred approach)',
            '- Endoscopic endonasal approach for tumor removal',
            '- Microscopic or endoscopic techniques',
            '- Preservation of normal pituitary tissue when possible',

            '**3. Medical Therapy**',
            '- Dopamine agonists (cabergoline, bromocriptine) for prolactinomas',
            '- Somatostatin analogs for growth hormone-secreting tumors',
            '- Hormone replacement therapy for deficiencies',
            '- Medical management of hormone excess states',

            '**4. Radiation Therapy**',
            '- Stereotactic radiosurgery for residual or recurrent tumors',
            '- Conventional radiation for aggressive tumors',
            '- Proton therapy for tumors near critical structures',

            '**5. Long-term Management**',
            '- Regular hormonal monitoring and replacement',
            '- Annual MRI surveillance',
            '- Visual field monitoring',
            '- Management of pituitary deficiencies',
            '- Fertility counseling if applicable'
        ],
        'duration': 'Recovery: 1-3 months, lifelong hormone management may be needed',
        'success_rate': '80-90% cure rate for most pituitary tumors'
    },
    'notumor': {
        'title': 'No Tumor Detected - Preventive Care Plan',
        'overview': 'No abnormalities detected in the MRI scan. Focus on preventive measures and general brain health.',
        'steps': [
            '**1. Confirmation and Documentation**',
            '- Review of imaging results by radiologist',
            '- Documentation of findings in medical records',
            '- Discussion of incidental findings if any',

            '**2. Preventive Measures**',
            '- Maintain healthy lifestyle (balanced diet, regular exercise)',
            '- Avoid smoking and excessive alcohol consumption',
            '- Regular cardiovascular health monitoring',
            '- Adequate sleep and stress management',

            '**3. Recommended Screenings**',
            '- Annual physical examination',
            '- Age-appropriate cancer screenings',
            '- Regular blood pressure and cholesterol monitoring',
            '- Vision and hearing assessments',

            '**4. Brain Health Maintenance**',
            '- Cognitive exercises and mental stimulation',
            '- Social engagement and community involvement',
            '- Mediterranean-style diet rich in antioxidants',
            '- Regular cardiovascular exercise',

            '**5. Follow-up Schedule**',
            '- Routine check-ups as recommended by primary care physician',
            '- Repeat MRI only if new symptoms develop',
            '- Monitoring of any pre-existing conditions',
            '- Health maintenance counseling'
        ],
        'duration': 'Ongoing preventive care with regular medical check-ups',
        'success_rate': 'Excellent prognosis with healthy lifestyle maintenance'
    }
}

def main():
    # Page Configuration
    st.set_page_config(page_title="Medical Disease Detection", layout="wide")
    
    st.title("Medical Disease Detection System")
    st.markdown("Use the sidebar to navigate and upload an image for analysis.")

    # Sidebar for Navigation
    st.sidebar.title("Options")
    app_mode = st.sidebar.selectbox("Choose Mode", ["Home", "Disease Detection", "About"])

    if app_mode == "Home":
        st.subheader("Welcome to the Medical Disease Detection System")
        st.write("This application uses Artificial Intelligence to detect diseases from medical imaging, specifically focusing on brain tumor detection from MRI scans.")
        
        st.markdown("### Key Features:")
        st.markdown("- **Advanced AI Model**: Utilizes deep learning algorithms trained on thousands of MRI images")
        st.markdown("- **Multi-Class Detection**: Identifies glioma, meningioma, pituitary tumors, and confirms no tumor presence")
        st.markdown("- **High Accuracy**: Provides confidence scores for reliable diagnosis assistance")
        st.markdown("- **Treatment Recommendations**: Offers preliminary treatment suggestions based on detected conditions")
        
        st.markdown("### How It Works:")
        st.markdown("1. Upload an MRI brain scan image in JPG, PNG, or JPEG format")
        st.markdown("2. Our AI model analyzes the image using convolutional neural networks")
        st.markdown("3. Receive instant results with tumor type classification and confidence level")
        st.markdown("4. Get recommended treatment options for detected conditions")
        
        st.info("Navigate to the **Disease Detection** page to start analyzing your medical images.")
        
        st.warning("**Important Disclaimer**: This tool is for educational and research purposes only. It should not replace professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.")

    elif app_mode == "Disease Detection":
        st.header("Upload Medical Image")
        
        # File Uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", width=300)
            
            # Predict Button
            if st.button("Predict"):
                with st.spinner("Analyzing..."):
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    # Preprocessing
                    IMAGE_SIZE = 128
                    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
                    img_array = img_to_array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Prediction
                    predictions = model.predict(img_array)
                    predicted_class_index = np.argmax(predictions, axis=1)[0]
                    confidence_score = np.max(predictions, axis=1)[0]
                    result = class_labels[predicted_class_index]

                    st.success("Analysis Complete!")
                    
                    if result == 'notumor':
                        st.write("### Prediction: **No Tumor Detected**")
                    else:
                        st.write(f"### Prediction: **Tumor Detected: {result}**")
                        
                    st.write(f"Confidence: **{confidence_score*100:.2f}%**")
                    st.progress(int(confidence_score * 100))
                    
                    # Display treatment
                    treatment_info = treatments[result]
                    st.write(f"### {treatment_info['title']}")
                    st.markdown(f"**Overview:** {treatment_info['overview']}")

                    st.markdown("#### Detailed Treatment Steps:")
                    for step in treatment_info['steps']:
                        st.markdown(step)

                    st.markdown(f"**Expected Duration:** {treatment_info['duration']}")
                    st.markdown(f"**Success Rate:** {treatment_info['success_rate']}")

                    st.warning("⚠️ **Important:** This is general information only. Treatment plans must be personalized by qualified medical professionals based on individual patient factors, tumor characteristics, and overall health status.")

    elif app_mode == "About":
        st.subheader("About the Medical Disease Detection Project")
        
        st.markdown("### Project Overview")
        st.write("This Medical Disease Detection System is an AI-powered application designed to assist in the preliminary analysis of brain MRI scans for tumor detection. The project demonstrates the application of deep learning in medical imaging and serves as an educational tool for understanding AI-assisted diagnostics.")
        
        st.markdown("### Technology Stack")
        st.markdown("- **Frontend**: Streamlit - Interactive web application framework")
        st.markdown("- **Backend**: Python with TensorFlow/Keras")
        st.markdown("- **AI Model**: Convolutional Neural Network (CNN) based on VGG16 architecture")
        st.markdown("- **Image Processing**: PIL (Python Imaging Library)")
        st.markdown("- **Data Science**: NumPy for numerical computations")
        
        st.markdown("### Model Details")
        st.markdown("- **Architecture**: Deep CNN with transfer learning from VGG16")
        st.markdown("- **Classes**: Glioma, Meningioma, Pituitary Tumor, No Tumor")
        st.markdown("- **Input Size**: 128x128 RGB images")
        st.markdown("- **Training Data**: Preprocessed MRI brain scan dataset")
        
        st.markdown("### Educational Purpose")
        st.write("This project was developed to:")
        st.markdown("- Demonstrate practical applications of deep learning in healthcare")
        st.markdown("- Provide an accessible interface for medical image analysis")
        st.markdown("- Serve as a learning resource for AI and medical imaging students")
        st.markdown("- Showcase the potential of AI in improving diagnostic workflows")
        
        st.markdown("### Important Notes")
        st.warning("This application is not intended for clinical use. All results should be verified by qualified medical professionals. The AI model provides probability-based predictions and should be used as a supplementary tool, not as a definitive diagnostic instrument.")
        
        st.markdown("### Future Enhancements")
        st.write("Potential improvements include:")
        st.markdown("- Integration with larger, more diverse datasets")
        st.markdown("- Multi-modal imaging support (CT, PET scans)")
        st.markdown("- Real-time model updates and continuous learning")
        st.markdown("- Integration with electronic health record systems")

if __name__ == "__main__":
    main()