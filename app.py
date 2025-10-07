import streamlit as st
import torch
from model import TransliterationModel
import time
import os

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
    .download-link {
        background-color: #4CAF50;
        color: white;
        padding: 10px 15px;
        text-decoration: none;
        border-radius: 5px;
        display: inline-block;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

def check_required_files():
    """Check if all required files exist"""
    required_files = {
        'Model file': 'best_model.pth',
        'Urdu vocabulary': 'trainingData/ur_vocab.txt',
        'English vocabulary': 'trainingData/en_vocab.txt'
    }
    
    missing_files = []
    for file_type, file_path in required_files.items():
        if not os.path.exists(file_path):
            missing_files.append((file_type, file_path))
    
    return missing_files

def download_vocab_files():
    """Provide instructions to download vocabulary files"""
    st.markdown("### 📥 Download Required Files")
    
    st.markdown("""
    Please download the following vocabulary files and place them in the `trainingData` directory:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Urdu Vocabulary:**
        1. Click the link below
        2. Right-click → "Save as"
        3. Save as `trainingData/ur_vocab.txt`
        """)
        st.markdown(
            '[Download ur_vocab.txt](https://raw.githubusercontent.com/Abdulbaset1/Urdu-to-Roman-Urdu-Translator/main/ur_vocab.txt)',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown("""
        **English Vocabulary:**
        1. Click the link below
        2. Right-click → "Save as"
        3. Save as `trainingData/en_vocab.txt`
        """)
        st.markdown(
            '[Download en_vocab.txt](https://raw.githubusercontent.com/Abdulbaset1/Urdu-to-Roman-Urdu-Translator/main/en_vocab.txt)',
            unsafe_allow_html=True
        )
    
    st.markdown("""
    ### 📁 Directory Structure
    After downloading, your project should look like this:
    ```
    your_project/
    ├── app.py
    ├── model.py
    ├── best_model.pth
    └── trainingData/
        ├── ur_vocab.txt
        └── en_vocab.txt
    ```
    """)

@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    try:
        # Check if files exist first
        missing_files = check_required_files()
        if missing_files:
            return None, missing_files
        
        model = TransliterationModel(
            model_path="best_model.pth",
            ur_vocab_path="trainingData/ur_vocab.txt",
            en_vocab_path="trainingData/en_vocab.txt",
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        return model, []
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, [("Model", f"Error: {str(e)}")]

def main():
    # Header
    st.markdown('<h1 class="main-header">🕌 Urdu to Roman Urdu Transliterator</h1>', unsafe_allow_html=True)
    
    # Check for missing files first
    missing_files = check_required_files()
    
    if missing_files:
        st.error("❌ Missing required files!")
        st.markdown("The following files are required but not found:")
        
        for file_type, file_path in missing_files:
            st.write(f"- **{file_type}**: `{file_path}`")
        
        st.markdown("---")
        download_vocab_files()
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
            help="Enter Urdu text in Urdu script"
        )
        
        # Transliterate button
        col1_1, col1_2, col1_3 = st.columns([1, 2, 1])
        with col1_2:
            transliterate_btn = st.button(
                "🔄 Transliterate", 
                type="primary", 
                use_container_width=True
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
        model, load_errors = load_model()
    
    if model is None:
        st.error("❌ Failed to load the model.")
        if load_errors:
            st.markdown("**Errors encountered:**")
            for error_type, error_msg in load_errors:
                st.write(f"- {error_type}: {error_msg}")
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
                st.error(f"❌ Error during transliteration: {e}")
    
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
