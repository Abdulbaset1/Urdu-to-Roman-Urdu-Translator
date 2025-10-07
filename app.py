import streamlit as st
import torch
from model import TransliterationModel
import time
import os
import requests
import urllib.request

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
        color: #000000;
        font-weight: bold;
    }
    .download-progress {
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def download_with_progress(url, filename):
    """Download file with progress bar"""
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            progress_bar.progress(percent)
            status_text.text(f"Downloading... {percent}% ({downloaded//1024}KB/{total_size//1024}KB)")
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        urllib.request.urlretrieve(url, filename, reporthook=progress_hook)
        
        progress_bar.empty()
        status_text.empty()
        return True
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        return False

def download_model_file():
    """Download the model file from GitHub releases"""
    model_url = "https://github.com/Abdulbaset1/Urdu-to-Roman-Urdu-Translator/releases/download/v.1/best_model.pth"
    model_path = "best_model.pth"
    
    if not os.path.exists(model_path):
        st.info("📥 Downloading trained model (best_model.pth)...")
        st.warning("This may take a while as the model file is large (~50MB)")
        
        if download_with_progress(model_url, model_path):
            st.success("✅ Model downloaded successfully!")
            return True
        else:
            st.error("❌ Failed to download model file")
            return False
    else:
        return True

def download_vocab_files():
    """Download vocabulary files from GitHub using raw URLs"""
    vocab_files = {
        'trainingData/ur_vocab.txt': 'https://raw.githubusercontent.com/Abdulbaset1/Urdu-to-Roman-Urdu-Translator/main/ur_vocab.txt',
        'trainingData/en_vocab.txt': 'https://raw.githubusercontent.com/Abdulbaset1/Urdu-to-Roman-Urdu-Translator/main/en_vocab.txt'
    }
    
    # Create trainingData directory if it doesn't exist
    os.makedirs('trainingData', exist_ok=True)
    
    success_count = 0
    
    for file_path, url in vocab_files.items():
        if not os.path.exists(file_path):
            try:
                st.info(f"📥 Downloading {file_path}...")
                response = requests.get(url)
                response.raise_for_status()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                success_count += 1
                st.success(f"✅ Successfully downloaded {file_path}")
                
            except Exception as e:
                st.error(f"❌ Failed to download {file_path}: {str(e)}")
                return False
        else:
            success_count += 1
    
    return success_count == len(vocab_files)

def check_vocab_files():
    """Check if vocabulary files are loaded correctly"""
    try:
        with open('trainingData/ur_vocab.txt', 'r', encoding='utf-8') as f:
            ur_lines = f.readlines()
        
        with open('trainingData/en_vocab.txt', 'r', encoding='utf-8') as f:
            en_lines = f.readlines()
        
        return True, len(ur_lines), len(en_lines)
    except Exception as e:
        st.error(f"Error reading vocab files: {e}")
        return False, 0, 0

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

def setup_files():
    """Download all required files only once"""
    # Check if all files already exist
    missing_files = check_required_files()
    if not missing_files:
        return True
    
    st.warning("⚠️ Setting up required files for the first time...")
    
    # Download model file
    model_success = download_model_file()
    if not model_success:
        return False
    
    # Download vocabulary files
    vocab_success = download_vocab_files()
    if not vocab_success:
        return False
    
    # Final check
    missing_files = check_required_files()
    if missing_files:
        st.error("❌ Failed to download all required files")
        return False
    
    return True

@st.cache_resource
def load_model():
    """Load the trained model with caching - this runs only once"""
    try:
        # Setup files first
        if not setup_files():
            return None
        
        # Verify vocab files can be read
        vocab_ok, ur_size, en_size = check_vocab_files()
        if not vocab_ok:
            st.error("❌ Failed to read vocabulary files")
            return None
        
        # Load the model
        with st.spinner("🔄 Loading model into memory..."):
            model = TransliterationModel(
                model_path="best_model.pth",
                ur_vocab_path="trainingData/ur_vocab.txt",
                en_vocab_path="trainingData/en_vocab.txt",
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        
        st.success("✅ Model loaded successfully! Ready for transliteration.")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🕌 Urdu to Roman Urdu Transliterator</h1>', unsafe_allow_html=True)
    
    # Load model only once when app starts
    model = load_model()
    
    if model is None:
        st.error("❌ Failed to load the model. Please check your internet connection and try again.")
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
        
        # Quick test button
        st.markdown("---")
        if st.button("🧪 Quick Test", key="test_button"):
            try:
                test_input = "تم"
                test_output = model.transliterate(test_input)
                st.success(f"Test: '{test_input}' → '{test_output}'")
            except Exception as e:
                st.error(f"Test failed: {str(e)}")

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
                        <small>⏱️ Processed in {processing_time:.3f} seconds</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Success message
                st.success("✅ Transliteration completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during transliteration: {str(e)}")
    
    elif transliterate_btn and not urdu_text.strip():
        st.warning("⚠️ Please enter some Urdu text to transliterate.")

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
