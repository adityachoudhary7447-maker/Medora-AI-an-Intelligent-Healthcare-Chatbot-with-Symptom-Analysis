#!/usr/bin/env python3
"""
SageMaker Inference Script for Medical Assistant Bot
Provides REST API endpoints for medical symptom analysis
"""

import os
import json
import sys
import logging
import traceback
from typing import Dict, Any, Optional
import base64
import io
from datetime import datetime

import flask
from flask import Flask, request, jsonify
from PIL import Image
import pandas as pd
import numpy as np

# Add project paths for imports
sys.path.append('/app')
sys.path.append('/app/src')

# Import medical assistant components
from src.medical_agent_langchain import AgenticMedicalAssistant
from src.medical_tools import analyze_symptoms_direct, analyze_combined_symptoms
from src.vision_tools import analyze_medical_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
app = Flask(__name__)
medical_agent = None
model_loaded = False

def load_model():
    """Load the medical assistant model and required resources"""
    global medical_agent, model_loaded
    
    try:
        logger.info("Loading medical assistant model...")
        
        # Initialize the medical agent
        medical_agent = AgenticMedicalAssistant()
        
        # Verify required files exist
        required_files = [
            '/app/indices/faiss_symptom_index_medibot/index.faiss',
            '/app/indices/faiss_severity_index_medibot/index.faiss',
            '/app/data/disease_symptoms.csv',
            '/app/data/disease_symptom_severity.csv'
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        model_loaded = True
        logger.info("Medical assistant model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def model_fn(model_dir: str):
    """SageMaker model loading function"""
    logger.info(f"Loading model from directory: {model_dir}")
    load_model()
    return medical_agent

def input_fn(request_body: str, content_type: str = 'application/json'):
    """Process input data for inference"""
    logger.info(f"Processing input with content type: {content_type}")
    
    try:
        if content_type == 'application/json':
            input_data = json.loads(request_body)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        return input_data
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        raise

def predict_fn(input_data: Dict[str, Any], model) -> Dict[str, Any]:
    """Main prediction function"""
    logger.info("Running prediction...")
    
    try:
        # Extract input parameters
        symptoms = input_data.get('symptoms', '')
        image_data = input_data.get('image')
        analysis_type = input_data.get('analysis_type', 'basic')
        
        # Validate input
        if not symptoms and not image_data:
            raise ValueError("Either symptoms text or image data must be provided")
        
        # Process image if provided
        image_analysis = None
        if image_data:
            try:
                # Decode base64 image
                if isinstance(image_data, str) and image_data.startswith('data:image'):
                    # Remove data URL prefix
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Analyze image
                image_analysis = analyze_medical_image(image)
                logger.info(f"Image analysis completed: {len(image_analysis)} symptoms found")
                
            except Exception as e:
                logger.error(f"Image processing error: {str(e)}")
                image_analysis = f"Error processing image: {str(e)}"
        
        # Perform medical analysis
        if analysis_type == 'comprehensive' and image_analysis:
            # Combined analysis with both text and image
            results = analyze_combined_symptoms(symptoms, image_analysis)
        else:
            # Basic symptom analysis
            results = analyze_symptoms_direct(symptoms)
        
        # Format response
        response = {
            'timestamp': datetime.utcnow().isoformat(),
            'analysis_type': analysis_type,
            'input_symptoms': symptoms,
            'image_analysis': image_analysis,
            'medical_analysis': results,
            'status': 'success'
        }
        
        logger.info("Prediction completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e),
            'status': 'error'
        }

def output_fn(prediction: Dict[str, Any], accept: str = 'application/json'):
    """Format output for response"""
    logger.info(f"Formatting output with accept type: {accept}")
    
    if accept == 'application/json':
        return json.dumps(prediction, indent=2)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

# Flask routes for direct API access
@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint"""
    try:
        status = 'healthy' if model_loaded else 'unhealthy'
        return jsonify({
            'status': status,
            'timestamp': datetime.utcnow().isoformat(),
            'model_loaded': model_loaded
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/invocations', methods=['POST'])
def invocations():
    """Main inference endpoint"""
    try:
        # Get request data
        content_type = request.content_type or 'application/json'
        
        # Process input
        input_data = input_fn(request.get_data(as_text=True), content_type)
        
        # Run prediction
        prediction = predict_fn(input_data, medical_agent)
        
        # Format output
        response = output_fn(prediction)
        
        return flask.Response(response, mimetype='application/json')
        
    except Exception as e:
        logger.error(f"Invocation error: {str(e)}")
        logger.error(traceback.format_exc())
        
        error_response = {
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'error'
        }
        
        return flask.Response(
            json.dumps(error_response),
            status=500,
            mimetype='application/json'
        )

@app.route('/health', methods=['GET'])
def health():
    """Extended health check with model status"""
    try:
        health_info = {
            'status': 'healthy' if model_loaded else 'initializing',
            'timestamp': datetime.utcnow().isoformat(),
            'model_loaded': model_loaded,
            'ollama_configured': bool(os.getenv('OLLAMA_BASE_URL')),
            'required_files': {}
        }
        
        # Check required files
        required_files = [
            '/app/indices/faiss_symptom_index_medibot/index.faiss',
            '/app/indices/faiss_severity_index_medibot/index.faiss',
            '/app/data/disease_symptoms.csv'
        ]
        
        for file_path in required_files:
            health_info['required_files'][file_path] = os.path.exists(file_path)
        
        return jsonify(health_info)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Start Flask server
    port = int(os.environ.get('SAGEMAKER_BIND_TO_PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)