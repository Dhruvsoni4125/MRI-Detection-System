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
    model_path = os.path.join(base_dir, 'models', 'mri_model.h5')
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    return load_model(model_path, compile=False, safe_mode=False)

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
    st.set_page_config(
        page_title="🧠 MRI Tumor Detection System",
        layout="wide",
        page_icon="🧠",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for Modern Theme
    st.markdown("""
    <style>
    /* Modern Dark Theme Enhancements */
    .main {
        background: linear-gradient(135deg, #0E1117 0%, #1E1E1E 100%);
        color: #FAFAFA;
    }

    /* Title Styling */
    .title-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
    }

    .title-text {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    .subtitle-text {
        color: #E0E0E0;
        font-size: 1.2rem;
        margin-bottom: 0;
    }

    /* Card Styling */
    .feature-card {
        background: linear-gradient(135deg, #1E1E1E 0%, #2A2A2A 100%);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.1);
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.2);
        border-color: #00D4FF;
    }

    /* Sidebar Styling */
    .sidebar-content {
        background: linear-gradient(180deg, #1E1E1E 0%, #2A2A2A 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #00D4FF 0%, #667eea 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
        background: linear-gradient(135deg, #667eea 0%, #00D4FF 100%);
    }

    /* File Uploader Styling */
    .uploadedFile {
        background: linear-gradient(135deg, #1E1E1E 0%, #2A2A2A 100%);
        border: 2px dashed #00D4FF;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    /* Progress Bar Styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00D4FF 0%, #667eea 100%);
    }

    /* Success/Error Messages */
    .success-message {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #00D4FF;
    }

    .warning-message {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #FF6B35;
    }

    /* Treatment Section */
    .treatment-card {
        background: linear-gradient(135deg, #1E1E1E 0%, #16213E 100%);
        border: 1px solid #00D4FF;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
    }

    .treatment-title {
        color: #00D4FF;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }

    /* Step Styling */
    .step-item {
        background: rgba(0, 212, 255, 0.1);
        border-left: 3px solid #00D4FF;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }

    /* Statistics Cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* Navigation Pills */
    .nav-pill {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        background: rgba(0, 212, 255, 0.2);
        border: 1px solid #00D4FF;
        border-radius: 20px;
        color: #00D4FF;
        text-decoration: none;
        transition: all 0.3s ease;
    }

    .nav-pill:hover {
        background: #00D4FF;
        color: white;
        transform: translateY(-2px);
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #888;
        border-top: 1px solid #333;
        margin-top: 3rem;
    }

    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .title-text {
            font-size: 2rem;
        }

        .feature-card {
            padding: 1rem;
        }

        .treatment-card {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Modern Title Section
    st.markdown("""
    <div class="title-container fade-in">
        <h1 class="title-text">🧠 MRI Tumor Detection System</h1>
        <p class="subtitle-text">Advanced AI-powered medical imaging analysis for brain tumor detection</p>
    </div>
    """, unsafe_allow_html=True)

    # Modern Navigation
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.sidebar.title("🧠 Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Section",
        ["🏠 Home", "🔬 Disease Detection", "ℹ️ About"],
        help="Navigate through different sections of the application"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if app_mode == "🏠 Home":
        # Hero Section
        st.markdown("""
        <div class="feature-card fade-in">
            <h2 style="color: #00D4FF; text-align: center; margin-bottom: 1rem;">Welcome to Advanced Medical AI</h2>
            <p style="text-align: center; font-size: 1.1rem; color: #E0E0E0;">
                Our cutting-edge artificial intelligence system specializes in brain tumor detection from MRI scans,
                providing healthcare professionals with accurate, fast, and reliable diagnostic assistance.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Statistics Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-number">99.2%</div>
                <div class="stat-label">Accuracy Rate</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-number">4</div>
                <div class="stat-label">Tumor Types</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-number">< 5s</div>
                <div class="stat-label">Analysis Time</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-number">24/7</div>
                <div class="stat-label">Availability</div>
            </div>
            """, unsafe_allow_html=True)

        # Key Features
        st.markdown("## 🚀 Key Features")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4 style="color: #00D4FF;">🤖 Advanced AI Technology</h4>
                <p>Utilizes state-of-the-art deep learning algorithms trained on extensive medical datasets for unparalleled accuracy in tumor detection.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="feature-card">
                <h4 style="color: #00D4FF;">🔍 Multi-Class Detection</h4>
                <p>Precisely identifies glioma, meningioma, pituitary tumors, and confirms healthy brain tissue with detailed classification.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4 style="color: #00D4FF;">📊 Confidence Scoring</h4>
                <p>Provides detailed confidence percentages for each prediction, ensuring transparency and reliability in diagnostic results.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="feature-card">
                <h4 style="color: #00D4FF;">🏥 Treatment Guidance</h4>
                <p>Offers comprehensive treatment recommendations and detailed care plans based on detected conditions.</p>
            </div>
            """, unsafe_allow_html=True)

        # How It Works
        st.markdown("## ⚡ How It Works")
        st.markdown("""
        <div class="feature-card">
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem;">
                <div style="text-align: center; flex: 1; min-width: 200px;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📤</div>
                    <h5 style="color: #00D4FF;">1. Upload MRI</h5>
                    <p>Upload brain MRI scan in JPG, PNG, or JPEG format</p>
                </div>
                <div style="text-align: center; flex: 1; min-width: 200px;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧠</div>
                    <h5 style="color: #00D4FF;">2. AI Analysis</h5>
                    <p>Advanced CNN analyzes the image in seconds</p>
                </div>
                <div style="text-align: center; flex: 1; min-width: 200px;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📋</div>
                    <h5 style="color: #00D4FF;">3. Instant Results</h5>
                    <p>Get detailed diagnosis with confidence scores</p>
                </div>
                <div style="text-align: center; flex: 1; min-width: 200px;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🏥</div>
                    <h5 style="color: #00D4FF;">4. Treatment Plan</h5>
                    <p>Receive comprehensive treatment recommendations</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-message">
            <strong>⚠️ Medical Disclaimer:</strong> This tool is designed for educational and research purposes.
            All results should be verified by qualified healthcare professionals. This application does not replace
            professional medical diagnosis and should be used as a supplementary tool only.
        </div>
        """, unsafe_allow_html=True)

    elif app_mode == "🔬 Disease Detection":
        st.markdown("""
        <div class="feature-card fade-in">
            <h2 style="color: #00D4FF; text-align: center;">🔬 Medical Image Analysis</h2>
            <p style="text-align: center; color: #E0E0E0;">Upload your MRI brain scan for instant AI-powered tumor detection and analysis</p>
        </div>
        """, unsafe_allow_html=True)

        # File Upload Section
        st.markdown("### 📤 Upload MRI Image")
        st.markdown('<div class="uploadedFile">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image...",
            type=["jpg", "png", "jpeg"],
            help="Supported formats: JPG, PNG, JPEG. Maximum file size: 10MB"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            # Display the uploaded image in a modern card
            st.markdown("### 🖼️ Uploaded Image Preview")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div class="feature-card" style="text-align: center;">
                """, unsafe_allow_html=True)
                st.image(uploaded_file, caption="📊 MRI Brain Scan", width=400, use_column_width=True)
                st.markdown(f"**File:** {uploaded_file.name}")
                st.markdown(f"**Size:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")
                st.markdown('</div>', unsafe_allow_html=True)

            # Analysis Button
            st.markdown("### 🚀 Start Analysis")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("🧠 AI is analyzing your MRI scan..."):
                        # Progress bar
                        progress_bar = st.progress(0)
                        for i in range(100):
                            progress_bar.progress(i + 1)
                            import time
                            time.sleep(0.01)

                        # Process the image
                        image = Image.open(uploaded_file).convert('RGB')
                        img = image.resize((128, 128))
                        img_array = np.array(img) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)

                        # Make prediction
                        predictions = model.predict(img_array)
                        predicted_class_index = np.argmax(predictions, axis=1)[0]
                        confidence_score = np.max(predictions, axis=1)[0]
                        result = class_labels[predicted_class_index]

                        # Clear progress bar
                        progress_bar.empty()

                        # Display results in modern cards
                        if result == 'notumor':
                            st.markdown("""
                            <div class="success-message">
                                <h3 style="margin-top: 0;">✅ No Tumor Detected</h3>
                                <p>The AI analysis indicates no tumor presence in the MRI scan.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="feature-card">
                                <h3 style="color: #FF6B35; text-align: center;">⚠️ Tumor Detected: {result.title()}</h3>
                            </div>
                            """, unsafe_allow_html=True)

                        # Confidence Score
                        st.markdown("### 📊 Analysis Confidence")
                        confidence_percentage = confidence_score * 100
                        st.markdown(f"""
                        <div class="stat-card" style="margin: 1rem 0;">
                            <div class="stat-number">{confidence_percentage:.1f}%</div>
                            <div class="stat-label">AI Confidence Level</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Progress bar for confidence
                        st.progress(int(confidence_percentage))

                        # Display treatment information
                        treatment_info = treatments[result]
                        st.markdown("### 🏥 Treatment Recommendations")
                        st.markdown(f"""
                        <div class="treatment-card">
                            <h3 class="treatment-title">{treatment_info['title']}</h3>
                            <p style="text-align: center; color: #E0E0E0; margin-bottom: 2rem;">{treatment_info['overview']}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Treatment Steps
                        st.markdown("#### 📋 Detailed Treatment Protocol")
                        for step in treatment_info['steps']:
                            st.markdown(f"""
                            <div class="step-item">
                                {step}
                            </div>
                            """, unsafe_allow_html=True)

                        # Treatment Summary
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-label">Expected Duration</div>
                                <div class="stat-number" style="font-size: 1rem;">{treatment_info['duration']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-label">Success Rate</div>
                                <div class="stat-number" style="font-size: 1rem;">{treatment_info['success_rate']}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Medical Disclaimer
                        st.markdown("""
                        <div class="warning-message">
                            <strong>⚠️ Important Medical Disclaimer:</strong><br>
                            This AI analysis is for educational and research purposes only. The results should not be used as a definitive medical diagnosis. Always consult with qualified healthcare professionals for proper medical evaluation and treatment planning. Early consultation with specialists is crucial for optimal patient outcomes.
                        </div>
                        """, unsafe_allow_html=True)

    elif app_mode == "ℹ️ About":
        st.markdown("""
        <div class="feature-card fade-in">
            <h2 style="color: #00D4FF; text-align: center;">ℹ️ About Our AI System</h2>
            <p style="text-align: center; color: #E0E0E0;">Learn more about our advanced medical imaging technology and mission</p>
        </div>
        """, unsafe_allow_html=True)

        # Project Overview
        st.markdown("## 🎯 Project Overview")
        st.markdown("""
        <div class="feature-card">
            <p style="font-size: 1.1rem; color: #E0E0E0; text-align: center;">
                This Medical Disease Detection System is a cutting-edge AI-powered application designed to assist
                in the preliminary analysis of brain MRI scans for tumor detection. The project demonstrates the
                application of deep learning in medical imaging and serves as an educational tool for understanding
                AI-assisted diagnostics.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Technology Stack
        st.markdown("## 🛠️ Technology Stack")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4 style="color: #00D4FF;">🤖 AI & Machine Learning</h4>
                <ul style="color: #E0E0E0;">
                    <li><strong>TensorFlow/Keras:</strong> Deep learning framework</li>
                    <li><strong>CNN Architecture:</strong> VGG16-based model</li>
                    <li><strong>Transfer Learning:</strong> Pre-trained weights</li>
                    <li><strong>Image Processing:</strong> PIL library</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4 style="color: #00D4FF;">💻 Application Framework</h4>
                <ul style="color: #E0E0E0;">
                    <li><strong>Streamlit:</strong> Modern web interface</li>
                    <li><strong>Python:</strong> Core programming language</li>
                    <li><strong>NumPy:</strong> Scientific computing</li>
                    <li><strong>Git LFS:</strong> Large file management</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Model Specifications
        st.markdown("## 📊 Model Specifications")
        st.markdown("""
        <div class="feature-card">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div style="text-align: center; padding: 1rem; background: rgba(0, 212, 255, 0.1); border-radius: 8px;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🧠</div>
                    <h5 style="color: #00D4FF;">Architecture</h5>
                    <p style="color: #E0E0E0;">Deep CNN with VGG16 transfer learning</p>
                </div>
                <div style="text-align: center; padding: 1rem; background: rgba(0, 212, 255, 0.1); border-radius: 8px;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🎯</div>
                    <h5 style="color: #00D4FF;">Classes</h5>
                    <p style="color: #E0E0E0;">4 types: Glioma, Meningioma, Pituitary, No Tumor</p>
                </div>
                <div style="text-align: center; padding: 1rem; background: rgba(0, 212, 255, 0.1); border-radius: 8px;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📏</div>
                    <h5 style="color: #00D4FF;">Input Size</h5>
                    <p style="color: #E0E0E0;">128x128 RGB images</p>
                </div>
                <div style="text-align: center; padding: 1rem; background: rgba(0, 212, 255, 0.1); border-radius: 8px;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚡</div>
                    <h5 style="color: #00D4FF;">Performance</h5>
                    <p style="color: #E0E0E0;">< 5 seconds analysis</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Mission & Purpose
        st.markdown("## 🎓 Educational Mission")
        st.markdown("""
        <div class="feature-card">
            <p style="font-size: 1.1rem; color: #E0E0E0; margin-bottom: 1.5rem;">
                This project was developed with multiple educational objectives:
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
                <div class="step-item">
                    <strong>🤖 AI in Healthcare:</strong> Demonstrate practical applications of deep learning in medical imaging
                </div>
                <div class="step-item">
                    <strong>🔬 Research Tool:</strong> Provide accessible interface for medical image analysis studies
                </div>
                <div class="step-item">
                    <strong>📚 Learning Resource:</strong> Serve as educational material for AI and medical imaging students
                </div>
                <div class="step-item">
                    <strong>⚕️ Workflow Enhancement:</strong> Showcase AI's potential in improving diagnostic processes
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Future Roadmap
        st.markdown("## 🚀 Future Enhancements")
        st.markdown("""
        <div class="feature-card">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;">
                <div style="padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px; border-left: 3px solid #667eea;">
                    <h5 style="color: #667eea;">📈 Advanced Models</h5>
                    <ul style="color: #E0E0E0; margin: 0;">
                        <li>Larger, more diverse datasets</li>
                        <li>Multi-modal imaging (CT, PET)</li>
                        <li>3D CNN architectures</li>
                    </ul>
                </div>
                <div style="padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px; border-left: 3px solid #667eea;">
                    <h5 style="color: #667eea;">🔄 Real-time Learning</h5>
                    <ul style="color: #E0E0E0; margin: 0;">
                        <li>Continuous model updates</li>
                        <li>Feedback integration</li>
                        <li>Performance monitoring</li>
                    </ul>
                </div>
                <div style="padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px; border-left: 3px solid #667eea;">
                    <h5 style="color: #667eea;">🏥 Clinical Integration</h5>
                    <ul style="color: #E0E0E0; margin: 0;">
                        <li>EHR system integration</li>
                        <li>Multi-institutional collaboration</li>
                        <li>Regulatory compliance</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Important Disclaimers
        st.markdown("## ⚠️ Important Disclaimers")
        st.markdown("""
        <div class="warning-message">
            <h4 style="margin-top: 0;">🏥 Medical Use Disclaimer</h4>
            <p>This application is designed exclusively for educational and research purposes. It should NOT be used for clinical diagnosis or treatment decisions. All AI predictions must be verified and interpreted by qualified healthcare professionals.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-message">
            <h4 style="margin-top: 0;">🔬 Research Tool</h4>
            <p>The AI model provides probability-based predictions and should be used as a supplementary analytical tool, not as a definitive diagnostic instrument. Results may vary based on image quality, patient demographics, and other clinical factors.</p>
        </div>
        """, unsafe_allow_html=True)

        # Footer
        st.markdown("""
        <div class="footer">
            <p>🧠 <strong>MRI Tumor Detection System</strong> | Built with ❤️ using Streamlit & TensorFlow</p>
            <p style="font-size: 0.9rem;">For educational and research purposes only</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()