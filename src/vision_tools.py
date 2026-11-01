"""
Updated Vision Tools for Medical Image Analysis
==============================================

Fixed version that properly gets image data from session state.
"""

import json
import re
import warnings
from typing import Callable, Optional
import base64

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain.tools import tool

warnings.filterwarnings('ignore')

# Global vision model and image data accessor
_VISION_MODEL = None
_IMAGE_DATA_ACCESSOR = None

def set_vision_model(vision_model: Optional[BaseChatModel]):
    """Set the global vision model for image analysis tools."""
    global _VISION_MODEL
    _VISION_MODEL = vision_model

def set_image_data_accessor(accessor_func: Callable):
    """Set the function to access current image data."""
    global _IMAGE_DATA_ACCESSOR
    _IMAGE_DATA_ACCESSOR = accessor_func
    print(f"🔧 Image data accessor set: {accessor_func is not None}")

def get_vision_model():
    """Get the current vision model."""
    return _VISION_MODEL

def get_current_image_data():
    """Get the current image data from session state."""
    if _IMAGE_DATA_ACCESSOR:
        try:
            data = _IMAGE_DATA_ACCESSOR()
            print(f"📷 Retrieved image data: {data is not None}")
            if data:
                print(f"📊 Image data length: {len(data)} characters")
            return data
        except Exception as e:
            print(f"❌ Error accessing image data: {e}")
            return None
    print("❌ No image data accessor configured")
    return None

def translate_medical_to_common_terms(technical_symptoms: list) -> list:
    """
    Translate technical medical terminology to common language that matches the FAISS database.
    """
    
    # Medical term to common language mapping
    medical_translation = {
        # Skin lesion types
        'papules': 'small bumps',
        'vesicles': 'small blisters',
        'pustules': 'pus-filled bumps',
        'macules': 'flat spots',
        'nodules': 'large bumps',
        'plaques': 'raised patches',
        'bullae': 'large blisters',
        'wheals': 'raised welts',
        'comedones': 'blackheads',
        'cysts': 'lumps under skin',
        
        # Color and appearance
        'erythematous': 'red',
        'erythema': 'redness',
        'hyperpigmented': 'dark spots',
        'hypopigmented': 'light spots',
        'purpuric': 'purple spots',
        'cyanotic': 'blue discoloration',
        'pallor': 'pale skin',
        'jaundice': 'yellow skin',
        
        # Texture and characteristics
        'scaling': 'flaky skin',
        'crusting': 'scabbed',
        'lichenification': 'thick skin',
        'atrophic': 'thin skin',
        'hyperkeratotic': 'thick rough skin',
        'desquamation': 'peeling',
        'excoriation': 'scratches',
        
        # Distribution and pattern
        'localized': 'in one area',
        'generalized': 'all over',
        'linear': 'in lines',
        'annular': 'ring-shaped',
        'targetoid': 'target-like',
        'reticular': 'net-like pattern',
        
        # Specific features
        'umbilicated': 'with central depression',
        'pedunculated': 'on a stalk',
        'sessile': 'flat-based',
        'keratotic': 'rough bumps',
        'follicular': 'around hair follicles',
        
        # Inflammation
        'inflammatory': 'inflamed',
        'edematous': 'swollen',
        'induration': 'hard area',
        'fluctuant': 'soft swelling',
        'tender': 'painful to touch',
        'pruritic': 'itchy',
        
        # Common conditions shortcuts
        'comedonal': 'blackheads and whiteheads',
        'acneiform': 'acne-like bumps',
        'eczematous': 'dry itchy patches',
        'psoriasiform': 'scaly red patches',
        'seborrheic': 'oily scaly patches'
    }
    
    translated_symptoms = []
    
    for symptom in technical_symptoms:
        symptom = symptom.strip().lower()
        
        # Direct translation if exact match
        if symptom in medical_translation:
            translated_symptoms.append(medical_translation[symptom])
            continue
        
        # Partial matching for compound terms
        translated_parts = []
        symptom_words = symptom.split()
        
        for word in symptom_words:
            if word in medical_translation:
                translated_parts.append(medical_translation[word])
            else:
                # Keep common words as-is
                if word not in ['lesions', 'areas', 'regions', 'patches', 'spots']:
                    translated_parts.append(word)
        
        if translated_parts:
            translated_symptoms.append(' '.join(translated_parts))
        else:
            # Fallback: try to simplify technical terms
            simplified = symptom.replace('lesions', 'spots').replace('areas', '').replace('regions', '')
            translated_symptoms.append(simplified.strip())
    
    # Remove duplicates while preserving order
    seen = set()
    final_symptoms = []
    for symptom in translated_symptoms:
        if symptom and symptom not in seen:
            seen.add(symptom)
            final_symptoms.append(symptom)
    
    return final_symptoms

def validate_base64_image(image_base64: str) -> dict:
    """
    Validate and process base64 image data with enhanced checks.
    """
    try:
        if not image_base64:
            return {'valid': False, 'error': 'No image data provided'}
        
        # Remove data URL prefix if present
        if image_base64.startswith('data:'):
            if ',' in image_base64:
                image_base64 = image_base64.split(',', 1)[1]
        
        # Clean up any whitespace or newlines
        image_base64 = image_base64.strip().replace('\n', '').replace('\r', '').replace(' ', '')
        
        # Validate base64 encoding
        try:
            decoded = base64.b64decode(image_base64, validate=True)
        except Exception as e:
            return {'valid': False, 'error': f'Invalid base64 encoding: {str(e)}'}
        
        # Check minimum size
        if len(decoded) < 1000:
            return {'valid': False, 'error': f'Image too small: {len(decoded)} bytes'}
        
        # Check maximum size
        if len(decoded) > 20 * 1024 * 1024:
            return {'valid': False, 'error': f'Image too large: {len(decoded)} bytes'}
        
        # Check image format by magic bytes
        image_format = 'unknown'
        if decoded.startswith(b'\xFF\xD8\xFF'):
            image_format = 'jpeg'
        elif decoded.startswith(b'\x89PNG\r\n\x1a\n'):
            image_format = 'png'
        elif decoded.startswith(b'GIF87a') or decoded.startswith(b'GIF89a'):
            image_format = 'gif'
        elif decoded.startswith(b'RIFF') and b'WEBP' in decoded[:12]:
            image_format = 'webp'
        elif decoded.startswith(b'BM'):
            return {'valid': False, 'error': 'BMP format is not supported by the configured vision model'}
        elif decoded.startswith(b'\xFF\xE0') or decoded.startswith(b'\xFF\xE1'):
            image_format = 'jpeg'
        
        if image_format == 'unknown':
            return {'valid': False, 'error': 'Unknown or unsupported image format'}
        
        # Additional validation for JPEG images
        if image_format == 'jpeg':
            if not (b'\xFF\xD9' in decoded[-10:]):
                return {'valid': False, 'error': 'Corrupted JPEG file - missing end marker'}
        
        return {
            'valid': True,
            'format': image_format,
            'size': len(decoded),
            'base64_clean': image_base64,
            'error': None
        }
        
    except Exception as e:
        return {'valid': False, 'error': f'Image validation failed: {str(e)}'}

@tool
def analyze_medical_image(additional_context: str = "") -> str:
    """
    Analyze the currently uploaded medical image to extract visible symptoms.
    Gets image data from session state instead of parameters.
    
    Args:
        additional_context: Optional additional context about the image
        
    Returns:
        JSON string containing extracted visible symptoms for medical analysis
    """
    print(f"🔍 ANALYZE_MEDICAL_IMAGE CALLED with context: '{additional_context}'")
    
    if not _VISION_MODEL:
        result = {
            "visible_symptoms": [],
            "analysis_summary": "Vision analysis is not available - no local Ollama vision model is configured",
            "success": False,
            "message": "Vision model not available"
        }
        print("❌ No vision model available")
        return json.dumps(result)
    
    # Get image data from session state via accessor
    image_base64 = get_current_image_data()
    if not image_base64:
        result = {
            "visible_symptoms": [],
            "analysis_summary": "No image data available in session state",
            "success": False,
            "message": "No image data available"
        }
        print("❌ No image data retrieved from session")
        return json.dumps(result)
    
    print(f"✅ Got image data from session: {len(image_base64)} characters")
    
    # Validate image data
    validation = validate_base64_image(image_base64)
    if not validation['valid']:
        result = {
            "visible_symptoms": [],
            "analysis_summary": f"Invalid image data: {validation['error']}",
            "success": False,
            "message": "Invalid image data"
        }
        print(f"❌ Image validation failed: {validation['error']}")
        return json.dumps(result)
    
    print(f"✅ Image validation passed: {validation['format']}, {validation['size']} bytes")
    
    try:
        # Determine MIME type based on format
        mime_type = "image/jpeg"  # Default
        if validation['format'] == 'png':
            mime_type = "image/png"
        elif validation['format'] == 'gif':
            mime_type = "image/gif"
        elif validation['format'] == 'webp':
            mime_type = "image/webp"
        
        # Focused prompt to extract symptoms in COMMON LANGUAGE
        focused_prompt = """You are a medical AI analyzing an image to extract VISIBLE SYMPTOMS using SIMPLE, COMMON LANGUAGE.

Your task: Identify visible medical symptoms using everyday terms that patients would use.

Focus ONLY on visible symptoms using SIMPLE WORDS:
- Skin conditions: Use "red spots", "bumps", "blisters", "rash", "swelling"
- Colors: Use "red", "dark spots", "light patches", "purple marks"  
- Textures: Use "rough skin", "flaky", "scaly", "smooth", "bumpy"
- Shapes: Use "round spots", "oval patches", "lines", "clusters"
- Size: Use "small", "large", "tiny", "big"

AVOID technical medical terms like:
- papules, vesicles, macules, nodules
- erythematous, hyperpigmented, lichenification
- inflammatory, keratotic, comedonal

GOOD examples:
- "small red bumps, blisters, swelling"
- "red rash, small spots, itchy-looking skin"  
- "round red patches, flaky skin"

BAD examples:
- "erythematous papules, vesicular lesions"
- "inflammatory comedonal acne"
- "lichenified plaques"

Respond with a simple comma-separated list using everyday language that a patient would understand."""

        context_addition = f"\n\nPatient context: {additional_context}" if additional_context else ""
        full_prompt = focused_prompt + context_addition
        
        # Create message for vision model
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": full_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{validation['base64_clean']}"
                        },
                    },
                ]
            )
        ]
        
        print("🚀 Calling local Ollama vision model...")
        
        # Retry transient local model failures without sending image data elsewhere.
        max_retries = 2
        response_text = None
        for attempt in range(max_retries + 1):
            try:
                response = _VISION_MODEL.invoke(messages)
                response_text = response.content.strip()
                print(f"✅ Local vision response received: {len(response_text)} characters")
                break
                
            except Exception as api_error:
                error_str = str(api_error).lower()
                print(f"❌ Local vision error (attempt {attempt + 1}): {api_error}")
                
                if "unsupported image" in error_str or "image_parse_error" in error_str:
                    result = {
                        "visible_symptoms": [],
                        "analysis_summary": "The local vision model rejected this image format.",
                        "success": False,
                        "message": "Image format not accepted by local vision model"
                    }
                    return json.dumps(result)
                        
                else:
                    if attempt < max_retries:
                        continue
                    else:
                        result = {
                            "visible_symptoms": [],
                            "analysis_summary": f"Local vision model error: {str(api_error)}",
                            "success": False,
                            "message": "Local vision inference failed"
                        }
                        return json.dumps(result)
        
        if not response_text:
            result = {
                "visible_symptoms": [],
                "analysis_summary": "No response received from the local Ollama vision model",
                "success": False,
                "message": "No API response"
            }
            return json.dumps(result)
        
        print(f"📝 Processing response: '{response_text[:100]}...'")
        
        # Extract symptoms from response
        cleaned_response = response_text.lower()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "visible symptoms:",
            "symptoms:",
            "i can see:",
            "the image shows:",
            "visible:",
            "observed:",
            "analysis:"
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned_response.startswith(prefix):
                cleaned_response = cleaned_response[len(prefix):].strip()
        
        # Split into individual symptoms
        raw_symptoms = []
        if cleaned_response:
            symptom_parts = re.split(r'[,;]|and(?=\s)', cleaned_response)
            
            for symptom in symptom_parts:
                symptom = symptom.strip()
                symptom = re.sub(r'^[-•*\d.\s]+', '', symptom)
                symptom = symptom.strip()
                
                if symptom and len(symptom) > 2 and symptom not in ['etc', 'other', 'some', 'any']:
                    raw_symptoms.append(symptom)
        
        print(f"🔍 Extracted raw symptoms: {raw_symptoms}")
        
        # Check if response contains technical medical terms
        technical_terms = ['papules', 'vesicles', 'erythematous', 'macules', 'nodules', 'lesions']
        has_technical_terms = any(term in cleaned_response for term in technical_terms)
        
        if has_technical_terms:
            symptoms = translate_medical_to_common_terms(raw_symptoms)
            print(f"🔄 Translated to common terms: {symptoms}")
        else:
            symptoms = raw_symptoms
        
        # Limit to most relevant symptoms
        symptoms = symptoms[:8]
        
        result = {
            "visible_symptoms": symptoms,
            "analysis_summary": f"Extracted {len(symptoms)} visible symptoms from {validation['format']} image analysis",
            "raw_response": response_text,
            "original_symptoms": raw_symptoms,
            "translation_applied": has_technical_terms,
            "image_info": {
                "format": validation['format'],
                "size_bytes": validation['size'],
                "mime_type": mime_type
            },
            "success": True,
            "message": f"Successfully extracted {len(symptoms)} visible symptoms"
        }
        
        print(f"✅ Analysis complete. Found symptoms: {symptoms}")
        return json.dumps(result)
        
    except Exception as e:
        print(f"❌ Error during image analysis: {e}")
        result = {
            "visible_symptoms": [],
            "analysis_summary": f"Error during image analysis: {str(e)}",
            "success": False,
            "message": f"Image analysis failed: {str(e)}"
        }
        return json.dumps(result)

@tool  
def analyze_visual_symptoms(visual_symptoms_list: str) -> str:
    """
    Analyze a list of visual symptoms to provide medical context.
    
    Args:
        visual_symptoms_list: Comma-separated list of visual symptoms
        
    Returns:
        JSON string with symptom analysis
    """
    if not visual_symptoms_list or not visual_symptoms_list.strip():
        result = {
            "symptom_analysis": "No visual symptoms provided for analysis",
            "medical_relevance": {},
            "success": False,
            "message": "No symptoms to analyze"
        }
        return json.dumps(result)
    
    symptoms = [s.strip() for s in visual_symptoms_list.split(',') if s.strip()]
    
    # Basic symptom categorization
    symptom_categories = {
        "inflammatory": ["erythema", "redness", "swelling", "edema", "inflammation"],
        "lesions": ["papules", "vesicles", "pustules", "nodules", "macules", "patches"],
        "texture": ["scaling", "crusting", "rough", "smooth", "bumpy"],
        "color": ["hyperpigmentation", "hypopigmentation", "discoloration", "pallor"],
        "distribution": ["localized", "generalized", "linear", "clustered", "scattered"]
    }
    
    analysis = {}
    for symptom in symptoms:
        symptom_lower = symptom.lower()
        categories = []
        
        for category, keywords in symptom_categories.items():
            if any(keyword in symptom_lower for keyword in keywords):
                categories.append(category)
        
        analysis[symptom] = {
            "categories": categories if categories else ["unclassified"],
            "medical_significance": "visible symptom requiring medical evaluation"
        }
    
    result = {
        "symptom_analysis": f"Analyzed {len(symptoms)} visual symptoms",
        "medical_relevance": analysis,
        "total_symptoms": len(symptoms),
        "success": True,
        "message": f"Visual symptom analysis completed for {len(symptoms)} symptoms"
    }
    
    return json.dumps(result)

def get_vision_tools():
    """
    Get enhanced vision tools for the medical assistant.
    
    Returns:
        List of vision tools
    """
    return [
        analyze_medical_image,
        analyze_visual_symptoms
    ]