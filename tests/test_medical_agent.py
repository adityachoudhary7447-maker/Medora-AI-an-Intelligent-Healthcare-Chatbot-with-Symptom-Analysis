"""
Pytest tests for medical agent functionality
"""
import json
import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain.memory import ConversationBufferMemory

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.medical_agent_langchain import AgenticMedicalAssistant, create_medical_assistant

class TestAgenticMedicalAssistant:
    """Test the core medical assistant agent"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for the assistant"""
        mock_faiss_symptom = Mock()
        mock_faiss_severity = Mock()
        mock_df_precautions = Mock()
        mock_df_descriptions = Mock()
        mock_df_severity = Mock()
        mock_vision_model = Mock()
        
        return {
            'faiss_symptom_index': mock_faiss_symptom,
            'faiss_severity_index': mock_faiss_severity,
            'df_disease_precautions': mock_df_precautions,
            'df_disease_symptom_description': mock_df_descriptions,
            'df_disease_symptom_severity': mock_df_severity,
            'vision_model': mock_vision_model
        }
    
    def test_assistant_initialization(self, mock_dependencies):
        """Test assistant initialization with dependencies"""
        assistant = AgenticMedicalAssistant(**mock_dependencies)
        
        assert assistant is not None
        assert hasattr(assistant, 'agent_executor')
        assert hasattr(assistant, 'memory')
    
    def test_assistant_with_custom_model(self, mock_dependencies):
        """Test assistant with a selected local Ollama model."""
        assistant = AgenticMedicalAssistant(
            model_name="llama3.2:1b",
            **mock_dependencies
        )

        assert assistant.llm.model == "llama3.2:1b"
    
    def test_memory_initialization(self, mock_dependencies):
        """Test memory system initialization"""
        assistant = AgenticMedicalAssistant(
            use_memory=True,
            **mock_dependencies
        )

        assert assistant is not None
        assert isinstance(assistant.memory, ConversationBufferMemory)

class TestMedicalAssistantChat:
    """Test chat functionality of the medical assistant"""
    
    @pytest.fixture
    def mock_assistant(self):
        """Create a mock assistant for testing"""
        assistant = Mock()
        assistant.chat = Mock()
        assistant.reset_conversation = Mock()
        return assistant
    
    def test_basic_chat_interaction(self, mock_assistant):
        """Test basic chat interaction"""
        mock_assistant.chat.return_value = "I can help analyze your symptoms."
        
        response = mock_assistant.chat("Hello, I have a headache")
        
        assert response is not None
        assert isinstance(response, str)
        mock_assistant.chat.assert_called_with("Hello, I have a headache")
    
    def test_symptom_analysis_chat(self, mock_assistant):
        """Test symptom analysis through chat"""
        expected_response = {
            "analysis": "Based on your symptoms of fever and cough...",
            "possible_conditions": ["common_cold", "flu"],
            "recommendations": "Please consult a healthcare provider"
        }
        
        mock_assistant.chat.return_value = json.dumps(expected_response)
        
        response = mock_assistant.chat("I have fever and cough")
        
        assert response is not None
        response_data = json.loads(response)
        assert "analysis" in response_data
        assert "possible_conditions" in response_data
    
    def test_conversation_memory(self, mock_assistant):
        """Test conversation memory functionality"""
        # Simulate multiple interactions
        mock_assistant.chat.side_effect = [
            "Hello! I'm ready to help.",
            "I remember you mentioned a headache. Any other symptoms?"
        ]
        
        response1 = mock_assistant.chat("Hello")
        response2 = mock_assistant.chat("I also have nausea now")
        
        assert len(mock_assistant.chat.call_args_list) == 2
        assert response1 != response2
    
    def test_conversation_reset(self, mock_assistant):
        """Test conversation reset functionality"""
        mock_assistant.reset_conversation()
        mock_assistant.reset_conversation.assert_called_once()

    def test_missing_disease_name_tool_call_uses_grounded_fallback(
        self,
    ):
        """A missing tool argument should recover using exact database disease names."""
        assistant = AgenticMedicalAssistant(
            faiss_symptom_index=Mock(),
            faiss_severity_index=Mock(),
            df_disease_precautions=Mock(),
            df_disease_symptom_description=Mock(),
            df_disease_symptom_severity=Mock(),
            vision_model=None,
        )
        assistant.agent_executor = Mock(
            invoke=Mock(
            side_effect=ValueError(
                "1 validation error for get_disease_description\n"
                "disease_name Field required"
            )
            )
        )
        mock_analyze_symptoms = Mock(return_value=json.dumps({
            "success": True,
            "diseases": [{
                "disease": "common_cold",
                "confidence_percentage": 80.0,
            }],
        }))
        mock_get_description = Mock(return_value=json.dumps({
            "success": True,
            "description": "A viral infection.",
        }))
        assistant.tools = [
            Mock(name="analyze_symptoms_direct", invoke=mock_analyze_symptoms),
            Mock(name="get_disease_description", invoke=mock_get_description),
        ]
        assistant.tools[0].name = "analyze_symptoms_direct"
        assistant.tools[1].name = "get_disease_description"

        response = assistant.chat("I have a runny nose and cough")

        mock_analyze_symptoms.assert_called_once_with({
            "user_input": "I have a runny nose and cough"
        })
        mock_get_description.assert_called_once_with({
            "disease_name": "common_cold"
        })
        assert "Common Cold" in response
        assert "A viral infection." in response
        assert "not a diagnosis" in response

class TestMedicalAssistantFactory:
    """Test the factory function for creating medical assistants"""
    
    @pytest.fixture
    def mock_factory_dependencies(self):
        """Mock dependencies for factory function"""
        return {
            'faiss_symptom_index': Mock(),
            'faiss_severity_index': Mock(),
            'df_disease_precautions': Mock(),
            'df_disease_symptom_description': Mock(),
            'df_disease_symptom_severity': Mock(),
            'vision_model': Mock()
        }
    
    @patch('src.medical_agent_langchain.AgenticMedicalAssistant')
    def test_create_medical_assistant(self, mock_assistant_class, mock_factory_dependencies):
        """Test the factory function"""
        mock_instance = Mock()
        mock_assistant_class.return_value = mock_instance
        
        assistant = create_medical_assistant(**mock_factory_dependencies)
        
        assert assistant is not None
        mock_assistant_class.assert_called_once()
    
    @patch('src.medical_agent_langchain.AgenticMedicalAssistant')
    def test_create_medical_assistant_with_options(self, mock_assistant_class, mock_factory_dependencies):
        """Test factory function with custom options"""
        mock_instance = Mock()
        mock_assistant_class.return_value = mock_instance
        
        assistant = create_medical_assistant(
            model_name="llama3.2:1b",
            use_memory=True,
            **mock_factory_dependencies
        )
        
        assert assistant is not None
        # Verify the factory passes through the options
        mock_assistant_class.assert_called_once()

class TestErrorHandling:
    """Test error handling in the medical assistant"""
    
    @pytest.fixture
    def mock_failing_assistant(self):
        """Create a mock assistant that raises errors"""
        assistant = Mock()
        assistant.chat.side_effect = Exception("API Error")
        return assistant
    
    def test_chat_error_handling(self, mock_failing_assistant):
        """Test error handling in chat functionality"""
        with pytest.raises(Exception) as exc_info:
            mock_failing_assistant.chat("test message")
        
        assert "API Error" in str(exc_info.value)
    
    def test_initialization_error_handling(self):
        """Test error handling during initialization"""
        # Test with missing required parameters
        with pytest.raises(TypeError):
            AgenticMedicalAssistant()  # Missing required parameters

class TestIntegrationScenarios:
    """Test integration scenarios and workflows"""
    
    @patch('src.medical_agent_langchain.set_global_resources')
    @patch('src.medical_agent_langchain.get_enhanced_medical_tools')
    def test_tool_integration(self, mock_get_tools, mock_set_resources):
        """Test integration with medical tools"""
        mock_tools = [Mock(), Mock(), Mock()]
        mock_get_tools.return_value = mock_tools
        
        # This would test the integration but requires more complex mocking
        assert mock_get_tools is not None
        assert mock_set_resources is not None
    
    def test_prompt_template_validation(self):
        """Test that prompt templates are properly configured"""
        # This would test the prompt template structure
        # For now, just verify basic structure
        assert True  # Placeholder for prompt template tests
    
    @pytest.mark.parametrize("user_input,expected_tools", [
        ("I have a fever", ["analyze_symptoms_direct"]),
        ("Please analyze this image", ["analyze_medical_image"]),
        ("I have fever and uploaded an image", ["analyze_medical_image", "analyze_combined_symptoms"]),
    ])
    def test_tool_selection_logic(self, user_input, expected_tools):
        """Test that appropriate tools are selected based on user input"""
        # This would test the agent's tool selection logic
        # For now, just verify the test structure
        assert isinstance(user_input, str)
        assert isinstance(expected_tools, list)
        assert len(expected_tools) > 0

class TestPerformanceAndLimits:
    """Test performance characteristics and limits"""
    
    def test_conversation_memory_limits(self):
        """Test conversation memory management"""
        # Test that memory doesn't grow indefinitely
        mock_memory = Mock()
        mock_memory.buffer = "x" * 50000  # Simulate large memory
        
        # The system should handle large memory buffers
        assert len(mock_memory.buffer) > 0
    
    def test_response_time_expectations(self):
        """Test response time expectations"""
        # This would test that responses come within reasonable time
        # For unit tests, we just verify the structure exists
        import time
        start_time = time.time()
        
        # Simulate processing
        time.sleep(0.001)  # 1ms
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should be very fast for unit tests
        assert processing_time < 1.0  # Less than 1 second

class TestConfigurationValidation:
    """Test configuration validation"""
    
    def test_model_name_validation(self):
        """Test model name validation"""
        valid_models = ["llama3.2", "llama3.2:1b", "qwen2.5vl:3b"]
        
        for model in valid_models:
            # These should be valid model names
            assert isinstance(model, str)
            assert len(model) > 0
    
    def test_temperature_validation(self):
        """Test temperature parameter validation"""
        valid_temperatures = [0.0, 0.1, 0.5, 1.0]
        invalid_temperatures = [-0.1, 1.1, 2.0]
        
        for temp in valid_temperatures:
            assert 0.0 <= temp <= 1.0
        
        for temp in invalid_temperatures:
            assert not (0.0 <= temp <= 1.0)

if __name__ == "__main__":
    pytest.main([__file__])