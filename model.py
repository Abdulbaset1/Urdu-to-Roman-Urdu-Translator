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
                response.raise_for_status()  # Raise an exception for bad status codes
                
                # Write the content to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                success_count += 1
                st.success(f"✅ Successfully downloaded {file_path}")
                
            except Exception as e:
                st.error(f"❌ Failed to download {file_path}: {str(e)}")
                # If download fails, create a basic version
                st.info(f"🔄 Creating basic {file_path}...")
                if create_basic_vocab_file(file_path):
                    success_count += 1
                    st.success(f"✅ Created basic {file_path}")
    
    return success_count == len(vocab_files)

def create_basic_vocab_file(file_path):
    """Create a basic vocabulary file if download fails"""
    try:
        if 'ur_vocab' in file_path:
            # Basic Urdu vocabulary
            basic_chars = [
                '<PAD>', '<UNK>', '<SOS>', '<EOS>',
                ' ', '!', '.', ',', '?',
                'ا', 'ب', 'پ', 'ت', 'ٹ', 'ث', 'ج', 'چ', 'ح', 'خ', 
                'د', 'ڈ', 'ذ', 'ر', 'ڑ', 'ز', 'ژ', 'س', 'ش', 'ص',
                'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل',
                'م', 'ن', 'و', 'ہ', 'ھ', 'ء', 'ی', 'ے',
                'آ', 'أ', 'ؤ', 'ئ', 'ة', 'ى',
                '۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹'
            ]
        else:
            # Basic English/Roman Urdu vocabulary
            basic_chars = [
                '<PAD>', '<UNK>', '<SOS>', '<EOS>',
                ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                ':', ';', '<', '=', '>', '?', '@',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                '[', '\\', ']', '^', '_', '`',
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                '{', '|', '}', '~'
            ]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for char in basic_chars:
                f.write(char + '\n')
        
        return True
    except Exception as e:
        st.error(f"❌ Failed to create {file_path}: {str(e)}")
        return False

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
        # First, try to download missing vocabulary files
        download_success = download_vocab_files()
        
        if not download_success:
            st.warning("Some vocabulary files might be incomplete, but continuing...")
        
        # Check if all files exist
        missing_files = check_required_files()
        if missing_files:
            st.error("❌ Missing required files:")
            for file in missing_files:
                st.write(f"- {file}")
            
            if 'best_model.pth' in missing_files:
                st.error("""
                **Critical: best_model.pth is missing!**
                
                Please ensure you have the trained model file. You need to:
                1. Train the model using your training code, OR
                2. Download the pre-trained model file
                
                Without best_model.pth, the app cannot function.
                """)
            return None
        
        # All files exist, load the model
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
        st.warning("⚠️ Setting up required files...")
        
        # Show what's missing
        st.write("**Missing files:**")
        for file in missing_files:
            st.write(f"- {file}")
        
        # Try to download/create missing vocab files
        if any('trainingData' in file for file in missing_files):
            st.info("🔄 Downloading vocabulary files...")
            download_vocab_files()
    
    # Check again after setup
    missing_files = check_required_files()
    if missing_files:
        st.error("❌ Still missing required files:")
        for file in missing_files:
            st.write(f"- {file}")
        
        if 'best_model.pth' in missing_files:
            st.error("""
            **You must have best_model.pth to continue!**
            
            Options:
            1. Train the model using your training code
            2. Use a pre-trained model file
            3. Check if best_model.pth is in the correct location
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
        
        # Vocabulary info
        try:
            with open('trainingData/ur_vocab.txt', 'r', encoding='utf-8') as f:
                ur_vocab_size = len(f.readlines())
            with open('trainingData/en_vocab.txt', 'r', encoding='utf-8') as f:
                en_vocab_size = len(f.readlines())
            
            st.write(f"**Urdu Vocab Size:** {ur_vocab_size}")
            st.write(f"**English Vocab Size:** {en_vocab_size}")
        except:
            st.write("**Vocab sizes:** Unable to read")

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
    model = load_model()
    
    if model is None:
        st.error("❌ Failed to load the model. Please check the errors above.")
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
    
    # Quick test button in sidebar
    with st.sidebar:
        st.markdown("---")
        if st.button("🧪 Test Model", key="test_button"):
            try:
                test_result = model.transliterate("تم")
                st.success(f"Test successful! 'تم' → '{test_result}'")
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
