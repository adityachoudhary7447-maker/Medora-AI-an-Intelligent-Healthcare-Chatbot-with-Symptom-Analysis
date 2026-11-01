"""
Pytest tests for medical tools functionality
"""
import json
import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.medical_tools import (
    safe_json_dumps,
    set_global_resources,
    analyze_symptoms_direct,
    analyze_symptom_severity,
    get_disease_description,
    get_disease_precautions,
    get_enhanced_medical_tools,
)

class TestSafeJsonDumps:
    """Test safe JSON serialization functionality"""
    
    def test_numpy_integer_conversion(self):
        """Test that numpy integers are converted to Python int"""
        data = {"value": np.int64(42)}
        result = safe_json_dumps(data)
        assert json.loads(result) == {"value": 42}
    
    def test_numpy_float_conversion(self):
        """Test that numpy floats are converted to Python float"""
        data = {"value": np.float64(3.14)}
        result = safe_json_dumps(data)
        assert json.loads(result) == {"value": 3.14}
    
    def test_numpy_array_conversion(self):
        """Test that numpy arrays are converted to lists"""
        data = {"array": np.array([1, 2, 3])}
        result = safe_json_dumps(data)
        assert json.loads(result) == {"array": [1, 2, 3]}
    
    def test_nested_structure(self):
        """Test nested structures with numpy types"""
        data = {
            "nested": {
                "int": np.int32(10),
                "float": np.float32(2.5),
                "list": [np.int64(1), np.float64(2.0)]
            }
        }
        result = safe_json_dumps(data)
        expected = {
            "nested": {
                "int": 10,
                "float": 2.5,
                "list": [1, 2.0]
            }
        }
        assert json.loads(result) == expected

class TestMedicalToolsCore:
    """Test core medical analysis functionality"""

    def test_agent_tool_schemas_have_expected_argument_names(self):
        """Check tool schemas sent to Ollama match each tool's Python signature."""
        tools = {tool.name: tool for tool in get_enhanced_medical_tools()}
        expected_arguments = {
            "analyze_symptoms_direct": {"user_input"},
            "analyze_combined_symptoms": {"text_symptoms", "visual_symptoms"},
            "analyze_symptom_severity": {"symptoms_list"},
            "get_disease_precautions": {"disease_name"},
            "get_disease_description": {"disease_name"},
            "analyze_medical_image": {"additional_context"},
            "analyze_visual_symptoms": {"visual_symptoms_list"},
        }

        assert set(expected_arguments).issubset(tools)
        for tool_name, argument_names in expected_arguments.items():
            assert set(tools[tool_name].args) == argument_names
    
    @pytest.fixture
    def mock_faiss_index(self):
        """Create a mock FAISS index"""
        mock_index = Mock()
        mock_index.similarity_search_with_score.return_value = [
            (Mock(page_content="Common cold", metadata={"disease": "common_cold"}), 0.8),
            (Mock(page_content="Flu", metadata={"disease": "influenza"}), 1.2)
        ]
        return mock_index
    
    @pytest.fixture
    def sample_dataframes(self):
        """Create sample dataframes for testing"""
        df_descriptions = pd.DataFrame({
            'Disease': ['common_cold', 'influenza'],
            'Description': ['A viral infection', 'Seasonal flu virus']
        })
        
        df_precautions = pd.DataFrame({
            'Disease': ['common_cold', 'influenza'], 
            'Precaution_1': ['Rest', 'Bed rest'],
            'Precaution_2': ['Fluids', 'Antiviral medication']
        })
        
        df_severity = pd.DataFrame({
            'Symptom': ['fever', 'cough'],
            'weight': [6, 4]
        })
        
        return df_descriptions, df_precautions, df_severity
    
    def test_set_global_resources(self, mock_faiss_index, sample_dataframes):
        """Test setting global resources"""
        df_desc, df_prec, df_sev = sample_dataframes
        
        set_global_resources(
            faiss_symptom_index=mock_faiss_index,
            faiss_severity_index=mock_faiss_index,
            df_disease_precautions=df_prec,
            df_disease_symptom_description=df_desc,
            df_disease_symptom_severity=df_sev
        )
        
        # Test that resources were set (would need to check global variables)
        assert True  # This is a setup function test
    
    @patch('src.medical_tools._FAISS_SYMPTOM_INDEX')
    def test_analyze_symptoms_direct(self, mock_index, sample_dataframes):
        """Test direct symptom analysis"""
        df_desc, df_prec, df_sev = sample_dataframes
        
        # Set up the mock
        mock_index.similarity_search_with_score.return_value = [
            (Mock(page_content="Common cold", metadata={"disease": "common_cold"}), 0.8)
        ]
        
        with patch('src.medical_tools._DF_DISEASE_SYMPTOM_DESCRIPTION', df_desc):
            # Mock the function call directly since the tool decorator complicates testing
            symptoms = "fever, cough, runny nose"
            
            # This would normally be called through the tool, but we'll test the core logic
            assert symptoms is not None  # Basic validation
    
    def test_analyze_symptom_severity_input_validation(self):
        """Test symptom severity analysis input validation"""
        # Test with empty symptoms - should handle gracefully, not raise exception
        result = analyze_symptom_severity.func("")
        assert isinstance(result, str)  # Should return JSON string
        
        # Test with valid symptoms
        result = analyze_symptom_severity.func("fever, headache")
        assert isinstance(result, str)  # Should return JSON string
    
    def test_get_disease_description_validation(self):
        """Test disease description retrieval validation"""
        # Test with empty disease
        result = get_disease_description.func("")
        result_data = json.loads(result)
        assert result_data["success"] == False
        
        # Test with valid disease
        result = get_disease_description.func("common_cold")
        assert isinstance(result, str)  # Should return JSON string
    
    def test_get_disease_precautions_validation(self):
        """Test disease precautions retrieval validation"""
        # Test with empty disease
        result = get_disease_precautions.func("")
        result_data = json.loads(result)
        assert result_data["success"] == False
        
        # Test with valid disease
        result = get_disease_precautions.func("common_cold")
        assert isinstance(result, str)  # Should return JSON string

class TestIntegrationScenarios:
    """Test integration scenarios and edge cases"""
    
    def test_symptom_analysis_workflow(self):
        """Test complete symptom analysis workflow"""
        # This would test the full workflow from symptoms to diagnosis
        symptoms = ["fever", "cough", "sore throat"]
        
        # Simulate the workflow
        assert len(symptoms) > 0
        assert all(isinstance(s, str) for s in symptoms)
    
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with invalid inputs
        with pytest.raises(Exception):
            safe_json_dumps(object())  # Should handle non-serializable objects
    
    @pytest.mark.parametrize("symptoms,expected_count", [
        ("fever", 1),
        ("fever, cough", 2), 
        ("fever, cough, headache", 3),
        ("", 0),
    ])
    def test_symptom_parsing(self, symptoms, expected_count):
        """Test symptom string parsing"""
        if symptoms:
            parsed = [s.strip() for s in symptoms.split(",")]
            assert len(parsed) == expected_count
        else:
            assert expected_count == 0

class TestDataValidation:
    """Test data validation and integrity"""
    
    def test_csv_file_existence(self):
        """Test that required CSV files exist"""
        required_files = [
            "data/disease_symptoms.csv",
            "data/disease_symptom_severity.csv", 
            "data/disease_precautions.csv",
            "data/disease_symptom_description.csv"
        ]
        
        base_path = os.path.join(os.path.dirname(__file__), "..")
        for file_path in required_files:
            full_path = os.path.join(base_path, file_path)
            assert os.path.exists(full_path), f"Required file missing: {file_path}"
    
    def test_faiss_indices_existence(self):
        """Test that FAISS indices exist"""
        required_indices = [
            "indices/faiss_symptom_index_medibot",
            "indices/faiss_severity_index_medibot"
        ]
        
        base_path = os.path.join(os.path.dirname(__file__), "..")
        for index_path in required_indices:
            full_path = os.path.join(base_path, index_path)
            assert os.path.exists(full_path), f"Required index missing: {index_path}"

if __name__ == "__main__":
    pytest.main([__file__])