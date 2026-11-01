#!/usr/bin/env python3
"""
Test script for SageMaker Medical Assistant Bot endpoint
"""

import json
import base64
import argparse
from typing import Dict, Any

import boto3
import requests
from botocore.exceptions import ClientError

def test_local_endpoint(url: str, test_data: Dict[str, Any]) -> Dict[str, Any]:
    """Test local Docker endpoint"""
    try:
        response = requests.post(
            f"{url}/invocations",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Local endpoint test failed: {str(e)}")
        return {'error': str(e)}

def test_sagemaker_endpoint(endpoint_name: str, test_data: Dict[str, Any], 
                          region: str = 'us-east-1') -> Dict[str, Any]:
    """Test SageMaker endpoint"""
    try:
        client = boto3.client('sagemaker-runtime', region_name=region)
        
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(test_data)
        )
        
        result = json.loads(response['Body'].read().decode())
        return result
    except ClientError as e:
        print(f"SageMaker endpoint test failed: {str(e)}")
        return {'error': str(e)}

def create_test_data() -> Dict[str, Any]:
    """Create test data for medical analysis"""
    return {
        'symptoms': 'I have a headache, fever, and sore throat',
        'analysis_type': 'basic'
    }

def create_image_test_data() -> Dict[str, Any]:
    """Create test data with image (placeholder)"""
    # This would normally include a base64-encoded medical image
    return {
        'symptoms': 'I have a rash on my arm',
        'analysis_type': 'comprehensive',
        'image': None  # Would include base64 image data
    }

def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test Medical Assistant Bot endpoints')
    
    parser.add_argument('--endpoint-type', choices=['local', 'sagemaker'], 
                       default='local', help='Type of endpoint to test')
    parser.add_argument('--url', default='http://localhost:8080',
                       help='URL for local endpoint')
    parser.add_argument('--endpoint-name', default='medical-assistant-endpoint',
                       help='SageMaker endpoint name')
    parser.add_argument('--region', default='us-east-1',
                       help='AWS region')
    parser.add_argument('--test-type', choices=['basic', 'image'], 
                       default='basic', help='Type of test to run')
    
    args = parser.parse_args()
    
    # Create test data
    if args.test_type == 'basic':
        test_data = create_test_data()
    else:
        test_data = create_image_test_data()
    
    print(f"Testing {args.endpoint_type} endpoint...")
    print(f"Test data: {json.dumps(test_data, indent=2)}")
    print("-" * 50)
    
    # Run test
    if args.endpoint_type == 'local':
        result = test_local_endpoint(args.url, test_data)
    else:
        result = test_sagemaker_endpoint(args.endpoint_name, test_data, args.region)
    
    # Display results
    print("Response:")
    print(json.dumps(result, indent=2))
    
    # Check for success
    if 'error' not in result and result.get('status') == 'success':
        print("\n✅ Test passed successfully!")
    else:
        print("\n❌ Test failed!")
        if 'error' in result:
            print(f"Error: {result['error']}")

if __name__ == '__main__':
    main()