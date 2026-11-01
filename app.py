"""
Medora AI: An Intelligent Healthcare Chatbot with Symptom Analysis Streamlit App
===================================================
.
"""

import os
import sys
import base64
import logging
from io import BytesIO
from typing import Optional

import streamlit as st
import pandas as pd
from PIL import Image, UnidentifiedImageError

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

# Add current directory and src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Page config
st.set_page_config(
    page_title="Medora AI: An Intelligent Healthcare Chatbot with Symptom Analysis",
    page_icon="🏥",
    layout="wide",
)

# Configure Streamlit file upload limits
if hasattr(st, "_config"):
    try:
        st._config.set_option("server.maxUploadSize", 1)  # 1MB limit
    except:
        pass

# CSS Styling
st.markdown(
    """
<style>
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;
        max-width: 100% !important;
    }
    
    .chat-messages {
        max-height: 400px;
        overflow-y: auto;
        border: 2px solid #333;
        border-radius: 10px;
        padding: 1rem;
        background: #1a1a1a;
        margin-bottom: 1rem;
    }
    
    .disclaimer {
        position: sticky;
        bottom: 0;
        background: var(--secondary-background-color);
        color: var(--text-color);
        padding: 1rem;
        text-align: center;
        border-top: 2px solid var(--primary-color);
        margin-top: 2rem;
        font-size: 0.85rem;
        border-radius: 8px 8px 0 0;
    }
    
    .stButton > button {
        width: 100%;
        margin-bottom: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Title
st.title("🏥 Medora AI: An Intelligent Healthcare Chatbot with Symptom Analysis")
st.markdown(
    "*AI-driven medical insights—built on multimodal data and collaborative reasoning.*"
)


def clear_all_data():
    """Clear all chat and image data."""
    st.session_state.messages = []
    st.session_state.uploaded_image = None
    st.session_state.image_base64 = None

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    st.session_state.uploader_key += 1

    if hasattr(st.session_state, "assistant") and st.session_state.assistant:
        st.session_state.assistant = None

    st.rerun()


def manage_conversation_memory():
    """Automatically manage conversation memory."""
    try:
        if len(st.session_state.messages) > 15:
            # Create a copy of messages to avoid dictionary changed size error
            messages_copy = list(st.session_state.messages)
            total_chars = sum(len(msg.get("content", "")) for msg in messages_copy)
            estimated_tokens = total_chars // 4

            if estimated_tokens > 100000:
                st.info("🔄 Auto-trimming old messages...")

                if len(messages_copy) > 10:
                    first_msg = messages_copy[0]
                    recent_msgs = messages_copy[-8:]

                    transition_msg = {
                        "role": "assistant",
                        "content": "📝 *[Earlier conversation trimmed - conversation continues below]*",
                        "has_image": False,
                    }

                    # Update session state with new list
                    st.session_state.messages = [
                        first_msg,
                        transition_msg,
                    ] + recent_msgs

                    if (
                        hasattr(st.session_state, "assistant")
                        and st.session_state.assistant
                        and hasattr(st.session_state.assistant, "reset_conversation")
                    ):
                        st.session_state.assistant.reset_conversation()
    except Exception as e:
        # Silently handle any memory management errors
        pass


def check_files():
    """Check if all required files are present."""
    required_files = {
        "Python modules": [
            "src/medical_tools.py",
            "src/medical_agent_langchain.py",
            "src/vision_tools.py",
        ],
        "Data files": [
            "data/disease_symptoms.csv",
            "data/disease_symptom_severity.csv",
            "data/disease_precautions.csv",
            "data/disease_symptom_description.csv",
        ],
        "FAISS indices": [
            "indices/faiss_symptom_index_medibot",
            "indices/faiss_severity_index_medibot",
        ],
    }

    missing_files = {}
    for file_category, files in required_files.items():
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            missing_files[file_category] = missing

    return missing_files


# Check files
missing = check_files()
if missing:
    st.error("⚠️ Missing required files!")
    for category, files in missing.items():
        st.write(f"**{category}:**")
        for f in files:
            st.write(f"  - {f}")
    st.stop()

# Try to import modules
try:
    from medical_agent_langchain import create_medical_assistant
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import ChatOllama
    from langchain.tools import tool
    from ollama_runtime import (
        OLLAMA_BASE_URL,
        OLLAMA_MODEL,
        OLLAMA_REQUEST_TIMEOUT,
        OLLAMA_VISION_MODEL,
        OllamaRuntimeError,
        format_ollama_error,
        require_model_available,
        require_vision_resources,
    )

    # Only show success message if there were previous errors or in debug mode
    # st.success("✅ All modules imported successfully")  # Hidden for clean UI

except ImportError as e:
    st.error(f"❌ Error importing modules: {str(e)}")
    st.stop()


# Create a custom tool for image analysis that accesses session state
@tool
def analyze_current_uploaded_image(additional_context: str = "") -> str:
    """
    Analyze the currently uploaded medical image in the Streamlit session.
    This tool accesses the image data from the current session state.

    Args:
        additional_context: Optional additional context about the image

    Returns:
        JSON string containing extracted visible symptoms for medical analysis
    """
    # Import vision tools
    try:
        import src.vision_tools as vision_tools

        # Get image data from session state
        image_data = getattr(st.session_state, "image_base64", None)

        if not image_data:
            return '{"visible_symptoms": [], "success": false, "message": "No image uploaded in current session"}'

        # Call the vision tool with the image data
        vision_tools.set_image_data_accessor(get_session_image_data)
        return vision_tools.analyze_medical_image.func(additional_context)

    except Exception as e:
        return f'{{"visible_symptoms": [], "success": false, "message": "Error analyzing image: {str(e)}"}}'


def encode_image_to_base64(image_file) -> Optional[str]:
    """Convert uploaded image to base64 string without processing."""
    try:
        if image_file is not None:
            image_file.seek(0)
            image_bytes = image_file.read()

            if not image_bytes:
                st.error("Image file appears to be empty")
                return None

            if len(image_bytes) < 1000:
                st.error("Image file is too small (less than 1KB)")
                return None

            if len(image_bytes) > 1 * 1024 * 1024:  # 1MB limit
                st.error(
                    "Image file is too large (over 1MB). Please compress the image or use a smaller file."
                )
                return None

            try:
                with Image.open(BytesIO(image_bytes)) as image:
                    if image.format not in {"PNG", "JPEG"}:
                        st.error(
                            "Unsupported image format. Please upload a PNG or JPEG image."
                        )
                        return None
                    image.verify()
            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
                Image.DecompressionBombError,
            ):
                st.error("The uploaded file is not a valid or supported image.")
                return None

            # Show file info
            file_size_kb = len(image_bytes) / 1024
            st.success(f"📁 Image loaded: {file_size_kb:.1f} KB")

            base64_string = base64.b64encode(image_bytes).decode("utf-8")
            st.success("✅ Image validated and ready for local analysis")
            return base64_string

        return None
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None


def display_image_preview(image_file, max_width=300):
    """Display image preview."""
    try:
        # Reset file pointer to beginning
        image_file.seek(0)

        # Open image once and reuse
        image = Image.open(image_file)
        original_width, original_height = image.size

        # Resize if needed
        if image.width > max_width:
            ratio = max_width / image.width
            new_height = int(image.height * ratio)
            image = image.resize((max_width, new_height))

        # Remove use_container_width parameter for compatibility
        st.image(image, caption="Uploaded Image", width=max_width)
        st.caption(f"Size: {original_width}x{original_height} px")

    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")
        print(f"Image display error: {e}")  # For debugging


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "assistant" not in st.session_state:
    st.session_state.assistant = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "image_base64" not in st.session_state:
    st.session_state.image_base64 = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    st.caption("Local inference through Ollama")
    try:
        require_model_available(OLLAMA_MODEL)
        st.success(f"✅ Ollama model installed: {OLLAMA_MODEL}")
    except OllamaRuntimeError as error:
        st.error(format_ollama_error(error, OLLAMA_MODEL))

    if OLLAMA_VISION_MODEL:
        try:
            require_vision_resources(OLLAMA_VISION_MODEL)
            st.success(f"✅ Local vision model ready: {OLLAMA_VISION_MODEL}")
        except OllamaRuntimeError as error:
            st.warning(f"Local image analysis is unavailable: {error}")
    else:
        st.info(
            "Image uploads are available. Local image analysis is disabled until "
            "OLLAMA_VISION_MODEL is set to an installed vision model with enough "
            "available RAM. Images are never sent to a cloud API."
        )

    st.markdown("---")

    st.subheader("📸 Image Analysis")

    uploaded_file = st.file_uploader(
        "Upload medical image (PNG/JPEG, max 1MB)",
        type=["png", "jpg", "jpeg"],
        help="Image stays on this computer. Maximum file size: 1MB.",
        key=f"image_uploader_{st.session_state.uploader_key}",
    )

    if uploaded_file is not None:
        st.session_state.uploaded_image = uploaded_file
        uploaded_file.seek(0)
        st.session_state.image_base64 = encode_image_to_base64(uploaded_file)

        if st.session_state.image_base64:
            display_image_preview(uploaded_file)
            st.info("💡 Image uploaded! Ask me to analyze it.")
    else:
        st.session_state.uploaded_image = None
        st.session_state.image_base64 = None
        st.info("Upload an image for AI-powered analysis")

    st.markdown("---")

    message_count = len(st.session_state.messages)
    if message_count > 0:
        try:
            # Create a copy to avoid iteration issues
            messages_copy = list(st.session_state.messages)
            total_chars = sum(len(msg.get("content", "")) for msg in messages_copy)
            estimated_tokens = total_chars // 4

            if estimated_tokens > 100000:
                st.warning(f"💬 Very long conversation ({message_count} messages)")
            elif estimated_tokens > 50000:
                st.info(f"💬 Long conversation ({message_count} messages)")
            else:
                st.caption(f"💬 {message_count} messages")
        except Exception:
            st.caption(f"💬 {message_count} messages")

    if st.button("🗑️ Clear Chat & Image", use_container_width=True):
        clear_all_data()


@st.cache_resource
def load_resources():
    """Load medical resources."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        symptom_index_path = "indices/faiss_symptom_index"
        if os.path.exists("indices/faiss_symptom_index_medibot"):
            symptom_index_path = "indices/faiss_symptom_index_medibot"

        severity_index_path = "indices/faiss_severity_index"
        if os.path.exists("indices/faiss_severity_index_medibot"):
            severity_index_path = "indices/faiss_severity_index_medibot"

        faiss_symptom = _load_faiss_index(symptom_index_path, embeddings)
        faiss_severity = _load_faiss_index(severity_index_path, embeddings)

        df_precautions = pd.read_csv("data/disease_precautions.csv")
        df_descriptions = pd.read_csv("data/disease_symptom_description.csv")
        df_severity = pd.read_csv("data/disease_symptom_severity.csv")

        return {
            "symptom_index": faiss_symptom,
            "severity_index": faiss_severity,
            "precautions": df_precautions,
            "descriptions": df_descriptions,
            "severity": df_severity,
        }
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        logger.error("Medical resource loading failed: %s", detail)
        st.error(f"Could not load medical resources. {detail}")
        return None


def _load_faiss_index(index_path: str, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Load an index and explain when a Git LFS pointer is used instead of data."""
    for filename in ("index.faiss", "index.pkl"):
        file_path = os.path.join(index_path, filename)
        with open(file_path, "rb") as index_file:
            header = index_file.readline(128)
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(
                f"'{file_path}' is an unresolved Git LFS pointer, not an index file. "
                "Run `git lfs pull` or regenerate the indexes with "
                "`python scripts/create_indices.py`."
            )

    return FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )


def get_session_image_data():
    """Get image data from Streamlit session state"""
    return getattr(st.session_state, "image_base64", None)


def init_assistant():
    """Initialize the assistant with local Ollama models and medical resources."""

    if st.session_state.assistant is None:
        try:
            require_model_available(OLLAMA_MODEL)
        except OllamaRuntimeError as error:
            st.error(format_ollama_error(error, OLLAMA_MODEL))
            return False

        with st.spinner("Loading medical knowledge base..."):
            resources = load_resources()

        if resources:
            try:
                import src.vision_tools as vision_tools

                vision_tools.set_image_data_accessor(get_session_image_data)

                vision_model = None
                if OLLAMA_VISION_MODEL:
                    try:
                        require_vision_resources(OLLAMA_VISION_MODEL)
                        vision_model = ChatOllama(
                            base_url=OLLAMA_BASE_URL,
                            model=OLLAMA_VISION_MODEL,
                            temperature=0.1,
                            timeout=OLLAMA_REQUEST_TIMEOUT,
                        )
                    except OllamaRuntimeError as error:
                        st.warning(f"Local image analysis is unavailable: {error}")

                st.session_state.assistant = create_medical_assistant(
                    faiss_symptom_index=resources["symptom_index"],
                    faiss_severity_index=resources["severity_index"],
                    df_disease_precautions=resources["precautions"],
                    df_disease_symptom_description=resources["descriptions"],
                    df_disease_symptom_severity=resources["severity"],
                    vision_model=vision_model,
                    model_name=OLLAMA_MODEL,
                    use_memory=True,
                )
                return True

            except Exception as e:
                detail = format_ollama_error(e, OLLAMA_MODEL)
                st.error(f"Failed to initialize the local assistant. {detail}")
                logger.error("Local assistant initialization failed: %s", detail)
                return False
    return True


def main_content():
    """Render main content with fixed image handling."""

    if not init_assistant():
        st.error(
            "❌ Assistant not initialized. Check the Ollama status and model configuration."
        )
        return

    st.markdown("### 💬 Chat History")

    if st.session_state.messages:
        try:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    if msg.get("has_image", False):
                        st.caption("📸 *Message includes uploaded image*")
                    st.write(msg.get("content", ""))
        except Exception:
            st.info(
                f"Chat history available ({len(st.session_state.messages)} messages)"
            )
    else:
        st.markdown(
            "### 👋 Welcome to Medora AI: An Intelligent Healthcare Chatbot with Symptom Analysis!"
        )
        st.markdown("""

        **🎯 To get started:**
        1. Describe your symptoms
        2. Upload an image (optional)
        3. Get AI analysis
        """)

    # Add image analysis button if image is uploaded (outside of columns)
    if st.session_state.uploaded_image and st.session_state.image_base64:
        if st.button("🔍 Analyze Uploaded Image", use_container_width=True):
            USER_PROMPT = "Please analyze the uploaded medical image."
        else:
            USER_PROMPT = None
    else:
        USER_PROMPT = None

    # Chat input must be at root level (not inside columns)
    if USER_PROMPT is None:
        USER_PROMPT = st.chat_input(
            'Describe symptoms or ask me to analyze the image (e.g. "I have a cough, fever and runny nose")...'
        )

    if USER_PROMPT:
        manage_conversation_memory()

        HAS_IMAGE = bool(st.session_state.image_base64)

        user_msg = {"role": "user", "content": USER_PROMPT, "has_image": HAS_IMAGE}
        st.session_state.messages.append(user_msg)

        with st.spinner("Analyzing..."):
            try:
                # Clean prompt without base64 data - the tools will get it from session
                if HAS_IMAGE and getattr(
                    st.session_state.assistant, "has_vision", False
                ):
                    enhanced_prompt = f"""
                    USER REQUEST: {USER_PROMPT}

                    INSTRUCTIONS: An image has been uploaded and is available to the local vision model.
                    1. Use analyze_medical_image() to extract visible symptoms from the uploaded image
                    2. If the user also provided text symptoms, use analyze_combined_symptoms() 
                    3. Always call get_disease_description() for diseases found
                    4. Use exact text from get_disease_description()
                    
                    The image must be analyzed locally; do not claim findings unless the vision tool succeeds.
                    """
                else:
                    enhanced_prompt = USER_PROMPT
                    if HAS_IMAGE:
                        enhanced_prompt += (
                            "\n\nA local vision model is not configured, so do not claim "
                            "to have analyzed the uploaded image. Respond only to any "
                            "text symptoms provided."
                        )

                if (
                    HAS_IMAGE
                    and not getattr(st.session_state.assistant, "has_vision", False)
                    and (
                        "analyze the uploaded medical image" in USER_PROMPT.lower()
                        or not USER_PROMPT.strip()
                    )
                ):
                    response = (
                        "Image analysis is currently unavailable because no compatible "
                        "local Ollama vision model is configured. The uploaded image was "
                        "not sent to a cloud API. Configure an installed local vision "
                        "model only when the computer has sufficient free memory."
                    )
                elif hasattr(st.session_state.assistant, "chat"):
                    response = st.session_state.assistant.chat(enhanced_prompt)
                else:
                    response = "Assistant not available"

                st.session_state.messages.append(
                    {"role": "assistant", "content": response, "has_image": HAS_IMAGE}
                )

                st.rerun()

            except Exception as e:
                error_str = str(e).lower()

                if "context_length_exceeded" in error_str:
                    ERROR_MSG = (
                        "🔄 Conversation too long. Please clear chat to continue."
                    )
                elif "rate limit" in error_str:
                    ERROR_MSG = "⏱️ Rate limit reached. Please wait and try again."
                elif "image" in error_str:
                    ERROR_MSG = "📸 Image processing issue. Try a different image."
                elif (
                    "ollama" in error_str
                    or "model" in error_str
                    or "memory" in error_str
                ):
                    ERROR_MSG = f"⚠️ {format_ollama_error(e, OLLAMA_MODEL)}"
                else:
                    ERROR_MSG = f"⚠️ Error: {str(e)[:200]}"

                st.session_state.messages.append(
                    {"role": "assistant", "content": ERROR_MSG, "has_image": False}
                )
                st.rerun()


# Run main content
main_content()

# Disclaimer
st.markdown(
    """
<div class="disclaimer">
⚠️ <strong>Disclaimer:</strong> This tool provides general information only. 
Always consult healthcare professionals for proper diagnosis and treatment.
</div>
""",
    unsafe_allow_html=True,
)
