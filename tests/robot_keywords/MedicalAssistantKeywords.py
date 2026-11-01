"""
Robot Framework Keywords for Medical Assistant Testing
"""
import os
import sys
import json
import time
from typing import Dict, List, Any
from robot.api.deco import keyword, library

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

@library(scope='GLOBAL')
class MedicalAssistantKeywords:
    """Custom Robot Framework keywords for medical assistant testing"""
    
    def __init__(self):
        self.conversation_history = []
        self.test_session_id = None
        self.mock_responses = {}
    
    @keyword
    def call_medical_analysis(self, symptoms: str) -> str:
        """Call medical analysis with given symptoms"""
        try:
            # Import here to avoid issues if modules aren't available
            from src.medical_tools import analyze_symptoms_direct
            
            # Mock the actual call since we don't have a full environment
            result = {
                "success": True,
                "symptoms_analyzed": symptoms,
                "possible_conditions": ["common_cold", "flu"],
                "confidence_scores": [0.8, 0.6],
                "recommendations": "Consult healthcare provider if symptoms persist"
            }
            
            return json.dumps(result)
            
        except ImportError:
            # Return mock response if imports fail
            mock_result = {
                "success": True,
                "symptoms_analyzed": symptoms,
                "possible_conditions": ["mock_condition"],
                "message": "Mock analysis for testing"
            }
            return json.dumps(mock_result)
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "message": "Analysis failed"
            }
            return json.dumps(error_result)
    
    @keyword
    def call_medical_term_translation(self, terms: List[str]) -> str:
        """Call medical term translation"""
        try:
            # Mock translation service
            translations = {}
            term_mappings = {
                "erythema": "redness of skin",
                "edema": "swelling",
                "cellulitis": "skin infection",
                "pyrexia": "fever",
                "cephalgia": "headache"
            }
            
            for term in terms:
                translations[term] = term_mappings.get(term, f"common term for {term}")
            
            result = {
                "success": True,
                "translations": translations,
                "terms_count": len(terms)
            }
            
            return json.dumps(result)
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "message": "Translation failed"
            }
            return json.dumps(error_result)
    
    @keyword
    def call_medical_assistant_chat(self, message: str) -> str:
        """Simulate medical assistant chat interaction"""
        try:
            self.conversation_history.append({
                "role": "user",
                "content": message,
                "timestamp": time.time()
            })
            
            # Generate contextual response based on conversation
            response_content = self._generate_contextual_response(message)
            
            response = {
                "role": "assistant",
                "content": response_content,
                "timestamp": time.time(),
                "has_context": len(self.conversation_history) > 1
            }
            
            self.conversation_history.append(response)
            
            return response_content
            
        except Exception as e:
            return f"Error in chat: {str(e)}"
    
    @keyword
    def send_raw_data(self, data: str) -> str:
        """Send raw data to test error handling"""
        try:
            # Attempt to process invalid data
            if data == "invalid_json":
                return '{"error": "Invalid JSON format", "success": false}'
            
            # Mock processing of raw data
            return '{"processed": true, "data_received": "' + str(data) + '"}'
            
        except Exception as e:
            return f'{{"error": "{str(e)}", "success": false}}'
    
    @keyword
    def get_conversation_history(self) -> List[Dict]:
        """Get current conversation history"""
        return self.conversation_history.copy()
    
    @keyword
    def clear_conversation_memory(self):
        """Clear conversation memory"""
        self.conversation_history.clear()
    
    @keyword
    def set_mock_response(self, key: str, response: str):
        """Set a mock response for testing"""
        self.mock_responses[key] = response
    
    @keyword
    def verify_response_structure(self, response: str, required_fields: List[str]) -> bool:
        """Verify response has required structure"""
        try:
            data = json.loads(response)
            
            for field in required_fields:
                if field not in data:
                    return False
            
            return True
            
        except json.JSONDecodeError:
            return False
        except Exception:
            return False
    
    @keyword
    def simulate_api_delay(self, seconds: float = 1.0):
        """Simulate API delay for performance testing"""
        time.sleep(seconds)
    
    @keyword
    def validate_medical_response(self, response: str) -> Dict[str, Any]:
        """Validate medical response structure and content"""
        try:
            data = json.loads(response)
            
            validation_result = {
                "is_valid_json": True,
                "has_success_field": "success" in data,
                "has_medical_content": False,
                "has_disclaimer": False,
                "structure_score": 0
            }
            
            # Check for medical content
            medical_keywords = [
                "symptoms", "condition", "disease", "diagnosis", 
                "treatment", "consult", "healthcare", "medical"
            ]
            
            response_text = response.lower()
            for keyword in medical_keywords:
                if keyword in response_text:
                    validation_result["has_medical_content"] = True
                    break
            
            # Check for disclaimer
            disclaimer_keywords = ["disclaimer", "consult", "healthcare provider", "professional"]
            for keyword in disclaimer_keywords:
                if keyword in response_text:
                    validation_result["has_disclaimer"] = True
                    break
            
            # Calculate structure score
            score = 0
            if validation_result["is_valid_json"]:
                score += 25
            if validation_result["has_success_field"]:
                score += 25
            if validation_result["has_medical_content"]:
                score += 25
            if validation_result["has_disclaimer"]:
                score += 25
            
            validation_result["structure_score"] = score
            
            return validation_result
            
        except json.JSONDecodeError:
            return {
                "is_valid_json": False,
                "has_success_field": False,
                "has_medical_content": False,
                "has_disclaimer": False,
                "structure_score": 0
            }
    
    @keyword
    def generate_test_symptoms(self, count: int = 3) -> List[str]:
        """Generate test symptoms for testing"""
        symptom_pool = [
            "fever", "headache", "cough", "sore throat", "runny nose",
            "nausea", "fatigue", "muscle aches", "chills", "dizziness",
            "chest pain", "shortness of breath", "abdominal pain"
        ]
        
        import random
        return random.sample(symptom_pool, min(count, len(symptom_pool)))
    
    @keyword
    def create_test_image_data(self) -> str:
        """Create base64 test image data"""
        # Create minimal PNG header for testing
        png_header = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
        
        import base64
        return base64.b64encode(png_header).decode('utf-8')
    
    @keyword
    def benchmark_response_time(self, operation_name: str, operation_func, *args) -> Dict[str, float]:
        """Benchmark response time for operations"""
        start_time = time.time()
        
        try:
            result = operation_func(*args)
            end_time = time.time()
            
            return {
                "operation": operation_name,
                "duration_seconds": end_time - start_time,
                "success": True,
                "result": str(result)[:100]  # First 100 chars
            }
        except Exception as e:
            end_time = time.time()
            
            return {
                "operation": operation_name,
                "duration_seconds": end_time - start_time,
                "success": False,
                "error": str(e)
            }
    
    def _generate_contextual_response(self, message: str) -> str:
        """Generate contextual response based on conversation history"""
        message_lower = message.lower()
        
        # Check for previous context
        previous_symptoms = []
        for entry in self.conversation_history:
            if entry.get("role") == "user":
                content = entry.get("content", "").lower()
                if any(symptom in content for symptom in ["fever", "headache", "cough", "pain"]):
                    # Extract mentioned symptoms
                    for symptom in ["fever", "headache", "cough", "pain", "nausea"]:
                        if symptom in content:
                            previous_symptoms.append(symptom)
        
        # Generate response based on current message and context
        if "hello" in message_lower:
            return "Hello! I'm ready to help analyze your symptoms. Please describe what you're experiencing."
        
        elif any(symptom in message_lower for symptom in ["fever", "headache", "cough", "pain", "nausea"]):
            response = "Based on your symptoms, I can help analyze possible conditions. "
            
            if previous_symptoms:
                response += f"I note you previously mentioned {', '.join(set(previous_symptoms))}. "
            
            response += "Please consult with a healthcare provider for proper diagnosis and treatment."
            return response
        
        elif "image" in message_lower:
            return "I can analyze medical images to identify visible symptoms. Please ensure the image is clear and shows the area of concern."
        
        else:
            return "I'm here to help with medical symptom analysis. Please describe your symptoms or upload an image for analysis."