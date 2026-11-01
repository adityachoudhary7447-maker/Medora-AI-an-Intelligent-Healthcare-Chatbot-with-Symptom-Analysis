#!/usr/bin/env python3
"""
SageMaker Deployment Script for Medical Assistant Bot
Deploys the containerized model to Amazon SageMaker
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Optional, Dict, Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Configuration
DEFAULT_INSTANCE_TYPE = 'ml.t2.medium'
DEFAULT_INSTANCE_COUNT = 1
DEFAULT_REGION = 'us-east-1'

def setup_logging():
    """Setup basic logging"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class SageMakerDeployer:
    """Handles SageMaker model deployment operations"""
    
    def __init__(self, region_name: str = DEFAULT_REGION):
        """Initialize SageMaker client"""
        try:
            self.region_name = region_name
            self.sagemaker_client = boto3.client('sagemaker', region_name=region_name)
            self.ecr_client = boto3.client('ecr', region_name=region_name)
            self.iam_client = boto3.client('iam', region_name=region_name)
            
            # Get account ID
            sts_client = boto3.client('sts', region_name=region_name)
            self.account_id = sts_client.get_caller_identity()['Account']
            
            logger.info(f"Initialized SageMaker deployer for region: {region_name}")
            logger.info(f"Account ID: {self.account_id}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS credentials.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to initialize SageMaker client: {str(e)}")
            sys.exit(1)
    
    def create_ecr_repository(self, repository_name: str) -> str:
        """Create ECR repository if it doesn't exist"""
        try:
            response = self.ecr_client.describe_repositories(
                repositoryNames=[repository_name]
            )
            repository_uri = response['repositories'][0]['repositoryUri']
            logger.info(f"ECR repository already exists: {repository_uri}")
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'RepositoryNotFoundException':
                logger.info(f"Creating ECR repository: {repository_name}")
                response = self.ecr_client.create_repository(
                    repositoryName=repository_name,
                    imageScanningConfiguration={'scanOnPush': True}
                )
                repository_uri = response['repository']['repositoryUri']
                logger.info(f"Created ECR repository: {repository_uri}")
            else:
                raise
        
        return repository_uri
    
    def get_execution_role(self, role_name: str = 'SageMakerExecutionRole') -> str:
        """Get or create SageMaker execution role"""
        try:
            response = self.iam_client.get_role(RoleName=role_name)
            role_arn = response['Role']['Arn']
            logger.info(f"Using existing execution role: {role_arn}")
            return role_arn
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchEntity':
                logger.info(f"Creating SageMaker execution role: {role_name}")
                return self.create_execution_role(role_name)
            else:
                raise
    
    def create_execution_role(self, role_name: str) -> str:
        """Create SageMaker execution role with required policies"""
        assume_role_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        try:
            # Create role
            response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy),
                Description='SageMaker execution role for Medical Assistant Bot'
            )
            role_arn = response['Role']['Arn']
            
            # Attach required policies
            policies = [
                'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
                'arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess'
            ]
            
            for policy_arn in policies:
                self.iam_client.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy_arn
                )
            
            # Wait for role to be available
            time.sleep(10)
            
            logger.info(f"Created execution role: {role_arn}")
            return role_arn
            
        except Exception as e:
            logger.error(f"Failed to create execution role: {str(e)}")
            raise
    
    def create_model(self, model_name: str, image_uri: str, role_arn: str) -> str:
        """Create SageMaker model"""
        try:
            # Delete existing model if it exists
            try:
                self.sagemaker_client.delete_model(ModelName=model_name)
                logger.info(f"Deleted existing model: {model_name}")
                time.sleep(5)
            except ClientError:
                pass
            
            # Create new model
            response = self.sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': image_uri,
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code',
                        'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                        'SAGEMAKER_REGION': self.region_name
                    }
                },
                ExecutionRoleArn=role_arn
            )
            
            model_arn = response['ModelArn']
            logger.info(f"Created SageMaker model: {model_arn}")
            return model_arn
            
        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")
            raise
    
    def create_endpoint_config(self, config_name: str, model_name: str, 
                             instance_type: str, instance_count: int) -> str:
        """Create endpoint configuration"""
        try:
            # Delete existing config if it exists
            try:
                self.sagemaker_client.delete_endpoint_config(
                    EndpointConfigName=config_name
                )
                logger.info(f"Deleted existing endpoint config: {config_name}")
                time.sleep(5)
            except ClientError:
                pass
            
            # Create new config
            response = self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': instance_count,
                        'InstanceType': instance_type,
                        'InitialVariantWeight': 1.0
                    }
                ]
            )
            
            config_arn = response['EndpointConfigArn']
            logger.info(f"Created endpoint configuration: {config_arn}")
            return config_arn
            
        except Exception as e:
            logger.error(f"Failed to create endpoint configuration: {str(e)}")
            raise
    
    def create_endpoint(self, endpoint_name: str, config_name: str) -> str:
        """Create or update SageMaker endpoint"""
        try:
            # Check if endpoint exists
            try:
                response = self.sagemaker_client.describe_endpoint(
                    EndpointName=endpoint_name
                )
                status = response['EndpointStatus']
                
                if status in ['InService', 'Creating', 'Updating']:
                    logger.info(f"Updating existing endpoint: {endpoint_name}")
                    self.sagemaker_client.update_endpoint(
                        EndpointName=endpoint_name,
                        EndpointConfigName=config_name
                    )
                else:
                    logger.info(f"Endpoint exists but in status: {status}")
                    return None
                    
            except ClientError as e:
                if e.response['Error']['Code'] == 'ValidationException':
                    # Endpoint doesn't exist, create it
                    logger.info(f"Creating new endpoint: {endpoint_name}")
                    response = self.sagemaker_client.create_endpoint(
                        EndpointName=endpoint_name,
                        EndpointConfigName=config_name
                    )
                else:
                    raise
            
            # Wait for endpoint to be in service
            logger.info("Waiting for endpoint to be in service...")
            waiter = self.sagemaker_client.get_waiter('endpoint_in_service')
            waiter.wait(
                EndpointName=endpoint_name,
                WaiterConfig={'Delay': 30, 'MaxAttempts': 60}
            )
            
            # Get endpoint ARN
            response = self.sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            endpoint_arn = response['EndpointArn']
            
            logger.info(f"Endpoint is in service: {endpoint_arn}")
            return endpoint_arn
            
        except Exception as e:
            logger.error(f"Failed to create/update endpoint: {str(e)}")
            raise
    
    def deploy(self, image_uri: str, endpoint_name: str, 
               instance_type: str = DEFAULT_INSTANCE_TYPE,
               instance_count: int = DEFAULT_INSTANCE_COUNT) -> Dict[str, str]:
        """Deploy model to SageMaker endpoint"""
        
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        model_name = f"medical-assistant-{timestamp}"
        config_name = f"medical-assistant-config-{timestamp}"
        
        logger.info(f"Starting deployment to endpoint: {endpoint_name}")
        logger.info(f"Image URI: {image_uri}")
        logger.info(f"Instance type: {instance_type}")
        logger.info(f"Instance count: {instance_count}")
        
        # Get execution role
        role_arn = self.get_execution_role()
        
        # Create model
        model_arn = self.create_model(model_name, image_uri, role_arn)
        
        # Create endpoint configuration
        config_arn = self.create_endpoint_config(
            config_name, model_name, instance_type, instance_count
        )
        
        # Create/update endpoint
        endpoint_arn = self.create_endpoint(endpoint_name, config_name)
        
        deployment_info = {
            'endpoint_name': endpoint_name,
            'endpoint_arn': endpoint_arn,
            'model_name': model_name,
            'model_arn': model_arn,
            'config_name': config_name,
            'config_arn': config_arn,
            'instance_type': instance_type,
            'instance_count': str(instance_count),
            'region': self.region_name,
            'deployment_time': timestamp
        }
        
        logger.info("Deployment completed successfully!")
        logger.info(f"Endpoint URL: https://runtime.sagemaker.{self.region_name}.amazonaws.com/endpoints/{endpoint_name}/invocations")
        
        return deployment_info

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='Deploy Medical Assistant Bot to SageMaker')
    
    parser.add_argument('--image-uri', required=True,
                       help='ECR image URI for the model')
    parser.add_argument('--endpoint-name', default='medical-assistant-endpoint',
                       help='SageMaker endpoint name')
    parser.add_argument('--instance-type', default=DEFAULT_INSTANCE_TYPE,
                       help='EC2 instance type for hosting')
    parser.add_argument('--instance-count', type=int, default=DEFAULT_INSTANCE_COUNT,
                       help='Number of instances for hosting')
    parser.add_argument('--region', default=DEFAULT_REGION,
                       help='AWS region for deployment')
    parser.add_argument('--create-ecr', action='store_true',
                       help='Create ECR repository if needed')
    
    args = parser.parse_args()
    
    try:
        deployer = SageMakerDeployer(region_name=args.region)
        
        # Create ECR repository if requested
        if args.create_ecr:
            repository_name = 'medical-assistant-bot'
            repository_uri = deployer.create_ecr_repository(repository_name)
            logger.info(f"ECR repository ready: {repository_uri}")
        
        # Deploy to SageMaker
        deployment_info = deployer.deploy(
            image_uri=args.image_uri,
            endpoint_name=args.endpoint_name,
            instance_type=args.instance_type,
            instance_count=args.instance_count
        )
        
        # Save deployment info
        output_file = f'deployment-info-{deployment_info["deployment_time"]}.json'
        with open(output_file, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info(f"Deployment information saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()