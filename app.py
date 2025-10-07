import streamlit as st
import torch
from model import TransliterationModel
import time
import os
import requests

# Page configuration
st.set_page_config(
    page_title="Urdu to Roman Urdu Transliterator",
    page_icon="🔤",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 1rem;
    }
    .input-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e6e6e6;
    }
    .urdu-text {
        font-size: 1.8rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: right;
        direction: rtl;
    }
    .roman-text {
        font-size: 1.5rem;
        font-family: 'Courier New', monospace;
        color: #d62728;
    }
</style>
""", unsafe_allow_html=True)

def download_vocab_files():
    """Download vocabulary files from GitHub using direct raw URLs"""
    # Create directory
    os.makedirs('trainingData', exist_ok=True)
    
    # GitHub raw URLs (direct download)
    vocab_files = {
        'trainingData/ur_vocab.txt': 'https://raw.githubusercontent.com/Abdulbaset1/Urdu-to-Roman-Urdu-Translator/main/ur_vocab.txt',
        'trainingData/en_vocab.txt': 'https://raw.githubusercontent.com/Abdulbaset1/Urdu-to-Roman-Urdu-Translator/main/en_vocab.txt'
    }
    
    downloaded_files = []
    
    for file_path, url in vocab_files.items():
        if not os.path.exists(file_path):
            try:
                st.info(f"📥 Downloading {file_path}...")
                response = requests.get(url)
                response.raise_for_status()  # Check for HTTP errors
                
                # Write content to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                downloaded_files.append(file_path)
                st.success(f"✅ Downloaded {file_path}")
                
            except Exception as e:
                st.error(f"❌ Failed to download {file_path}: {str(e)}")
                return False
    
    return True

def check_required_files():
    """Check if all required files exist"""
    required_files = [
        'best_model.pth',
        'trainingData/ur_vocab.txt',
        'trainingData/en_vocab.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    return missing_files

@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    try:
        # Download vocabulary files if they don't exist
        if not os.path.exists('trainingData/ur_vocab.txt') or not os.path.exists('trainingData/en_vocab.txt'):
            success = download_vocab_files()
            if not success:
                st.error("❌ Failed to download vocabulary files")
                return None
        
        # Check if model file exists
        if not os.path.exists('best_model.pth'):
            st.error("❌ Model file 'best_model.pth' not found!")
            return None
        
        # Load the model
        model = TransliterationModel(
            model_path="best_model.pth",
            ur_vocab_path="trainingData/ur_vocab.txt",
            en_vocab_path="trainingData/en_vocab.txt",
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        st.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🕌 Urdu to Roman Urdu Transliterator</h1>', unsafe_allow_html=True)
    
    # Check for required files
    missing_files = check_required_files()
    
    if missing_files:
        st.warning("⚠️ Some files are missing. Setting up...")
        
        # Download vocabulary files if missing
        if any('trainingData' in file for file in missing_files):
            st.info("🔄 Downloading vocabulary files from GitHub...")
            download_vocab_files()
        
        # Check again after download attempt
        missing_files = check_required_files()
    
    # Final check
    if missing_files:
        st.error("❌ Missing required files:")
        for file in missing_files:
            st.write(f"- {file}")
        
        if 'best_model.pth' in missing_files:
            st.error("""
            **Critical: best_model.pth is missing!**
            
            Please ensure you have the trained model file in the same directory as app.py
            """)
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.info("""
        This tool converts Urdu text to Roman Urdu (English script) 
        using a deep learning model based on LSTM encoder-decoder architecture.
        
        **How to use:**
        1. Type or paste Urdu text in the input box
        2. Click 'Transliterate' button
        3. View the Roman Urdu output
        """)
        
        st.markdown("### 📊 Model Info")
        st.write("**Architecture:** Seq2Seq with BiLSTM Encoder & LSTM Decoder")
        st.write("**Training:** 30 epochs on Urdu-Roman Urdu pairs")
        st.write("**Vocab Size:** Custom character-level vocabulary")
        
        # Device info
        device = "GPU 🚀" if torch.cuda.is_available() else "CPU ⚙️"
        st.write(f"**Running on:** {device}")
        
        # File status
        st.markdown("### ✅ File Status")
        st.success("All required files are present!")
        
        # Show vocabulary info
        try:
            with open('trainingData/ur_vocab.txt', 'r', encoding='utf-8') as f:
                ur_vocab_size = len(f.readlines())
            with open('trainingData/en_vocab.txt', 'r', encoding='utf-8') as f:
                en_vocab_size = len(f.readlines())
            
            st.write(f"**Urdu Vocab Size:** {ur_vocab_size}")
            st.write(f"**English Vocab Size:** {en_vocab_size}")
        except:
            pass

    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="input-box">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">📝 Input Urdu Text</h3>', unsafe_allow_html=True)
        
        # Text input
        urdu_text = st.text_area(
            "Enter Urdu text:",
            height=150,
            placeholder="Type or paste Urdu text here...\nExample: تم کون ہو",
            help="Enter Urdu text in Urdu script",
            key="urdu_input"
        )
        
        # Transliterate button
        col1_1, col1_2, col1_3 = st.columns([1, 2, 1])
        with col1_2:
            transliterate_btn = st.button(
                "🔄 Transliterate", 
                type="primary", 
                use_container_width=True,
                key="transliterate_btn"
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3 class="sub-header">🔤 Output Roman Urdu</h3>', unsafe_allow_html=True)
        
        # Result display area
        result_placeholder = st.empty()
        result_placeholder.markdown("""
        <div class="result-box">
            <p style='text-align: center; color: #666; font-style: italic;'>
                The transliterated text will appear here...
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Load model
    with st.spinner("🔄 Loading transliteration model..."):
        model = load_model()
    
    if model is None:
        st.error("❌ Failed to load the model. Please check if 'best_model.pth' exists.")
        return

    # Process transliteration when button is clicked
    if transliterate_btn and urdu_text.strip():
        with st.spinner("🔄 Transliterating..."):
            start_time = time.time()
            
            try:
                # Perform transliteration
                roman_output = model.transliterate(urdu_text.strip())
                processing_time = time.time() - start_time
                
                # Display results
                with col2:
                    result_placeholder.markdown(f"""
                    <div class="result-box">
                        <h4>📥 Input (Urdu):</h4>
                        <div class="urdu-text">{urdu_text.strip()}</div>
                        <hr>
                        <h4>📤 Output (Roman Urdu):</h4>
                        <div class="roman-text">{roman_output}</div>
                        <br>
                        <small>⏱️ Processed in {processing_time:.2f} seconds</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Success message
                st.success("✅ Transliteration completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during transliteration: {str(e)}")
    
    elif transliterate_btn and not urdu_text.strip():
        st.warning("⚠️ Please enter some Urdu text to transliterate.")
    
    # Example section
    st.markdown("---")
    st.markdown('<h3 class="sub-header">📚 Examples</h3>', unsafe_allow_html=True)
    
    examples_col1, examples_col2, examples_col3 = st.columns(3)
    
    example_pairs = [
        ("تم کون ہو", "tum kaun ho"),
        ("میں ٹھیک ہوں", "main theek hoon"),
        ("آپ کا نام کیا ہے", "aap ka naam kya hai")
    ]
    
    for i, (urdu, roman) in enumerate(example_pairs):
        with [examples_col1, examples_col2, examples_col3][i]:
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; text-align: center;">
                <div class="urdu-text" style="font-size: 1.2rem;">{urdu}</div>
                <div class="roman-text" style="font-size: 1rem;">{roman}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Test button in sidebar
    with st.sidebar:
        st.markdown("---")
        if st.button("🧪 Quick Test", key="test_button"):
            try:
                test_input = "تم"
                test_output = model.transliterate(test_input)
                st.success(f"Test: '{test_input}' → '{test_output}'")
            except Exception as e:
                st.error(f"Test failed: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using PyTorch & Streamlit | Urdu-Roman Urdu Transliteration Model"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
