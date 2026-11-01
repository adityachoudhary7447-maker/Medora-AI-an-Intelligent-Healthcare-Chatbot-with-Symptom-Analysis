"""
Updated Medical Tools with Vision Integration
============================================

This version integrates vision tools and focuses on core medical analysis functionality.
"""

import json
import re
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from langchain.tools import tool
from langchain_community.vectorstores import FAISS

# Import vision tools
try:
    from src.vision_tools import get_vision_tools, set_vision_model
    VISION_TOOLS_AVAILABLE = True
except ImportError:
    VISION_TOOLS_AVAILABLE = False

def safe_json_dumps(obj):
    """Safely serialize objects to JSON, converting numpy types to Python types."""
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj

    converted_obj = convert_types(obj)
    return json.dumps(converted_obj)

warnings.filterwarnings('ignore')

# ===============================
# GLOBAL VARIABLES FOR TOOLS
# ===============================
_FAISS_SYMPTOM_INDEX = None
_FAISS_SEVERITY_INDEX = None
_DF_DISEASE_PRECAUTIONS = None
_DF_DISEASE_SYMPTOM_DESCRIPTION = None
_DF_DISEASE_SYMPTOM_SEVERITY = None

def set_global_resources(faiss_symptom_index: FAISS,
                         faiss_severity_index: FAISS,
                         df_disease_precautions: pd.DataFrame,
                         df_disease_symptom_description: pd.DataFrame,
                         df_disease_symptom_severity: pd.DataFrame,
                         vision_model=None):
    """Set global resources for tools to access."""
    global _FAISS_SYMPTOM_INDEX, _FAISS_SEVERITY_INDEX
    global _DF_DISEASE_PRECAUTIONS
    global _DF_DISEASE_SYMPTOM_DESCRIPTION, _DF_DISEASE_SYMPTOM_SEVERITY

    print(f"🔧 Setting global resources...")
    print(f"📊 FAISS symptom index: {faiss_symptom_index is not None}")
    print(f"📊 FAISS severity index: {faiss_severity_index is not None}")
    print(f"👁️ Vision model: {vision_model is not None}")

    _FAISS_SYMPTOM_INDEX = faiss_symptom_index
    _FAISS_SEVERITY_INDEX = faiss_severity_index
    _DF_DISEASE_PRECAUTIONS = df_disease_precautions
    _DF_DISEASE_SYMPTOM_DESCRIPTION = df_disease_symptom_description
    _DF_DISEASE_SYMPTOM_SEVERITY = df_disease_symptom_severity
    
    # Set vision model in vision tools
    if VISION_TOOLS_AVAILABLE and vision_model:
        set_vision_model(vision_model)
        print("✅ Vision model set in vision tools")

def clean_user_input(user_input: str) -> str:
    """Lightly clean user input to remove conversation noise while preserving medical terms."""
    cleaned = user_input.lower().strip()
    
    conversation_patterns = [
        r'\b(hello|hi|hey)\b',
        r'\b(doctor|doc)\b', 
        r'\b(i think|i believe|i feel like)\b',
        r'\b(what should i do|can you help|please help)\b',
        r'\b(i have been|i am|i\'m)\s+(experiencing|having|feeling)\b'
    ]
    
    for pattern in conversation_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', cleaned)
    
    return cleaned

# ===============================
# CORE MEDICAL ANALYSIS TOOLS
# ===============================

@tool
def analyze_symptoms_direct(user_input: str) -> str:
    """
    Direct symptom analysis using FAISS without explicit symptom extraction.
    This is the main function for analyzing user-described symptoms.
    """
    if not _FAISS_SYMPTOM_INDEX:
        result = {
            "diseases": [],
            "extracted_symptoms": [],
            "success": False,
            "message": "Disease database not available"
        }
        return safe_json_dumps(result)

    if not user_input.strip():
        result = {
            "diseases": [],
            "extracted_symptoms": [],
            "success": False,
            "message": "No input provided"
        }
        return safe_json_dumps(result)

    cleaned_input = clean_user_input(user_input)

    try:
        results = _FAISS_SYMPTOM_INDEX.similarity_search_with_score(
            cleaned_input, k=15
        )

        disease_results = []
        found_symptoms = set()

        for doc, sim_score in results:
            sim_score = float(sim_score)

            if sim_score <= 1.5:
                disease_name = doc.metadata.get('disease', 'unknown')
                confidence_score = max(0.0, 1.0 - (sim_score / 2.0))
                confidence_percentage = round(confidence_score * 100, 1)
                
                matched_content = doc.page_content.lower()
                content_symptoms = [s.strip() for s in re.split(r'[,;]', matched_content) if s.strip()]
                found_symptoms.update(content_symptoms)

                result_dict = {
                    'disease': str(disease_name),
                    'similarity_score': sim_score,
                    'confidence_score': confidence_score,
                    'confidence_percentage': confidence_percentage,
                    'matched_symptoms': content_symptoms,
                    'matched_content': str(doc.page_content),
                    'all_metadata': doc.metadata
                }

                disease_results.append(result_dict)

        disease_results.sort(key=lambda x: -x['confidence_score'])

        seen_diseases = set()
        unique_results = []
        for result in disease_results:
            disease = result['disease']
            if disease not in seen_diseases:
                seen_diseases.add(disease)
                unique_results.append(result)

        extracted_symptoms = list(found_symptoms)

        result = {
            "diseases": unique_results[:5],
            "extracted_symptoms": extracted_symptoms,
            "total_matches": len(results),
            "good_matches": len(disease_results),
            "search_query": cleaned_input,
            "original_input": user_input,
            "success": True,
            "message": f"Found {len(unique_results[:5])} diseases via FAISS"
        }
        
        return safe_json_dumps(result)

    except Exception as e:
        result = {
            "diseases": [],
            "extracted_symptoms": [],
            "success": False,
            "message": f"Error in direct symptom analysis: {e}"
        }
        return safe_json_dumps(result)

@tool
def analyze_combined_symptoms(text_symptoms: str, visual_symptoms: str) -> str:
    """
    Analyze combined text and visual symptoms for comprehensive diagnosis.
    
    Args:
        text_symptoms: Comma-separated text symptoms from user input
        visual_symptoms: Comma-separated visual symptoms from image analysis
        
    Returns:
        JSON string containing combined analysis results
    """
    print(f"🔍 COMBINED ANALYSIS CALLED:")
    print(f"📝 Text symptoms: {text_symptoms}")
    print(f"👁️ Visual symptoms: {visual_symptoms}")
    
    # Combine all symptoms
    all_symptoms = []
    
    if text_symptoms and text_symptoms.strip():
        text_list = [s.strip() for s in text_symptoms.split(',') if s.strip()]
        all_symptoms.extend(text_list)
        print(f"📝 Added {len(text_list)} text symptoms")
    
    if visual_symptoms and visual_symptoms.strip():
        visual_list = [s.strip() for s in visual_symptoms.split(',') if s.strip()]
        all_symptoms.extend(visual_list)
        print(f"👁️ Added {len(visual_list)} visual symptoms")
    
    if not all_symptoms:
        result = {
            "diseases": [],
            "success": False,
            "message": "No symptoms provided for analysis"
        }
        print(f"❌ COMBINED ANALYSIS FAILED: No symptoms")
        return safe_json_dumps(result)
    
    # Use combined symptoms for analysis
    combined_symptoms_str = ', '.join(all_symptoms)
    print(f"🔄 Analyzing combined symptoms: {combined_symptoms_str}")
    
    result = analyze_symptoms_direct(combined_symptoms_str)
    
    # Parse and enhance result
    result_data = json.loads(result)
    if result_data.get("success"):
        result_data["analysis_type"] = "combined"
        result_data["text_symptoms_count"] = len([s.strip() for s in text_symptoms.split(',') if s.strip()]) if text_symptoms else 0
        result_data["visual_symptoms_count"] = len([s.strip() for s in visual_symptoms.split(',') if s.strip()]) if visual_symptoms else 0
        result_data["combined_symptoms"] = all_symptoms
        print(f"✅ COMBINED ANALYSIS SUCCESS: Found {len(result_data.get('diseases', []))} diseases")
    else:
        print(f"❌ COMBINED ANALYSIS FAILED: {result_data.get('message')}")
    
    return safe_json_dumps(result_data)

@tool
def analyze_symptom_severity(symptoms_list: str) -> str:
    """Analyzes the severity of symptoms using FAISS similarity and severity database."""
    if not _FAISS_SEVERITY_INDEX or _DF_DISEASE_SYMPTOM_SEVERITY is None:
        result = {
            "severity_scores": {},
            "overall_severity": 4.0,
            "overall_level": "Moderate",
            "success": False,
            "message": "Severity database not available"
        }
        return json.dumps(result)

    symptoms = [s.strip() for s in symptoms_list.split(',') if s.strip()]

    if not symptoms:
        result = {
            "severity_scores": {},
            "overall_severity": 4.0,
            "overall_level": "Moderate",
            "success": False,
            "message": "No symptoms provided"
        }
        return json.dumps(result)

    def classify_severity(severity_score: float) -> str:
        if severity_score <= 2:
            return "Low"
        elif severity_score <= 3:
            return "Mild"
        elif severity_score <= 4:
            return "Moderate"
        elif severity_score <= 5:
            return "High"
        elif severity_score <= 6:
            return "Very High"
        else:
            return "Severe"

    severity_scores = {}

    for symptom in symptoms:
        try:
            results = _FAISS_SEVERITY_INDEX.similarity_search_with_score(symptom, k=1)

            if results:
                matched_doc = results[0][0]
                matched_symptom = matched_doc.page_content.strip()

                exact_match = _DF_DISEASE_SYMPTOM_SEVERITY[
                    (_DF_DISEASE_SYMPTOM_SEVERITY['symptom'].str.lower().str.strip() == matched_symptom.lower())
                ]

                if len(exact_match) > 0:
                    severity_scores[symptom] = float(exact_match['severity'].iloc[0])
                else:
                    partial_matches = _DF_DISEASE_SYMPTOM_SEVERITY[
                        (_DF_DISEASE_SYMPTOM_SEVERITY['symptom'].str.lower().str.contains(matched_symptom.lower(), na=False))
                    ]

                    if len(partial_matches) > 0:
                        avg_severity = partial_matches['severity'].mean()
                        severity_scores[symptom] = float(avg_severity)
                    else:
                        severity_scores[symptom] = 4.0
            else:
                severity_scores[symptom] = 4.0

        except Exception:
            severity_scores[symptom] = 4.0

    if severity_scores:
        valid_scores = [score for score in severity_scores.values() if score is not None]
        if valid_scores:
            avg_severity = sum(valid_scores) / len(valid_scores)
            overall_level = classify_severity(avg_severity)
        else:
            avg_severity = 4.0
            overall_level = "Moderate"
    else:
        avg_severity = 4.0
        overall_level = "Moderate"

    result = {
        "severity_scores": {k: float(v) for k, v in severity_scores.items()},
        "overall_severity": float(avg_severity),
        "overall_level": overall_level,
        "success": True,
        "message": f"Analyzed severity for {len(symptoms)} symptoms"
    }
    
    return json.dumps(result)

@tool
def get_disease_precautions(disease_name: str) -> str:
    """Gets precautions and preventive measures for a specific disease."""
    if _DF_DISEASE_PRECAUTIONS is None:
        result = {
            "precautions": "Precautions database not available.",
            "success": False,
            "message": "Database not available"
        }
        return json.dumps(result)

    try:
        precautions = _DF_DISEASE_PRECAUTIONS[
            (_DF_DISEASE_PRECAUTIONS['disease'].str.lower() == disease_name.lower())
        ]['precautions'].iloc[0]

        result = {
            "precautions": str(precautions),
            "success": True,
            "message": f"Found precautions for {disease_name}"
        }
        return json.dumps(result)

    except (IndexError, KeyError) as e:
        result = {
            "precautions": "No specific precautions found for this condition.",
            "success": False,
            "message": f"No precautions found for {disease_name}"
        }
        return json.dumps(result)

@tool
def get_disease_description(disease_name: str) -> str:
    """Gets detailed description for a specific disease."""
    if _DF_DISEASE_SYMPTOM_DESCRIPTION is None:
        result = {
            "description": "Description database not available.",
            "success": False,
            "message": "Database not available"
        }
        return json.dumps(result)

    try:
        disease_col = None
        desc_col = None

        for col in _DF_DISEASE_SYMPTOM_DESCRIPTION.columns:
            if col.lower() in ['disease', 'diseases']:
                disease_col = col
                break

        for col in _DF_DISEASE_SYMPTOM_DESCRIPTION.columns:
            if col.lower() in ['description', 'desc', 'descriptions']:
                desc_col = col
                break

        if disease_col is None or desc_col is None:
            result = {
                "description": f"Unable to find appropriate columns. Available: {list(_DF_DISEASE_SYMPTOM_DESCRIPTION.columns)}",
                "success": False,
                "message": "Column structure issue"
            }
            return json.dumps(result)

        description = _DF_DISEASE_SYMPTOM_DESCRIPTION[
            (_DF_DISEASE_SYMPTOM_DESCRIPTION[disease_col].str.lower() == disease_name.lower())
        ][desc_col].iloc[0]

        result = {
            "description": str(description),
            "success": True,
            "message": f"Found description for {disease_name}"
        }
        return json.dumps(result)

    except (IndexError, KeyError) as e:
        result = {
            "description": f"No detailed description found for {disease_name}.",
            "success": False,
            "message": f"No description found for {disease_name}"
        }
        return json.dumps(result)

# ===============================
# TOOL UTILITIES
# ===============================

def get_enhanced_medical_tools():
    """
    Get all enhanced medical tools including vision tools.
    
    Returns:
        List of medical tools for LangChain agent
    """
    # Core medical tools
    medical_tools = [
        analyze_symptoms_direct,        # Text symptom analysis
        analyze_combined_symptoms,      # Combined text + visual analysis
        analyze_symptom_severity,       # Severity analysis
        get_disease_precautions,       # Precautions
        get_disease_description        # Disease descriptions
    ]
    
    # Add vision tools if available
    if VISION_TOOLS_AVAILABLE:
        vision_tools = get_vision_tools()
        medical_tools.extend(vision_tools)
    
    return medical_tools

def check_resources_available() -> Dict[str, bool]:
    """Check which resources are available."""
    resources = {
        'faiss_symptom_index': _FAISS_SYMPTOM_INDEX is not None,
        'faiss_severity_index': _FAISS_SEVERITY_INDEX is not None,
        'df_disease_precautions': _DF_DISEASE_PRECAUTIONS is not None,
        'df_disease_symptom_description': _DF_DISEASE_SYMPTOM_DESCRIPTION is not None,
        'df_disease_symptom_severity': _DF_DISEASE_SYMPTOM_SEVERITY is not None,
        'vision_tools_available': VISION_TOOLS_AVAILABLE
    }
    
    if VISION_TOOLS_AVAILABLE:
        from src.vision_tools import get_vision_model
        resources['vision_model'] = get_vision_model() is not None
    else:
        resources['vision_model'] = False
    
    return resources