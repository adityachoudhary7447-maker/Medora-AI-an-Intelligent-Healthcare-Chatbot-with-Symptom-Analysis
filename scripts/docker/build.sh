#!/bin/bash
# Docker Build Script for Medical Assistant Bot
set -e

# Configuration
IMAGE_NAME="medical-assistant-bot"
TAG=${1:-latest}
REGISTRY=${DOCKER_REGISTRY:-}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Move to project root
cd "$(dirname "$0")/../.."

print_status "Building Medical Assistant Bot Docker images..."

# Build production image
print_status "Building production image..."
docker build --target production -t "${IMAGE_NAME}:${TAG}" .

# Build SageMaker image
print_status "Building SageMaker image..."
docker build --target sagemaker -t "${IMAGE_NAME}-sagemaker:${TAG}" .

# Tag images for registry if specified
if [ -n "$REGISTRY" ]; then
    print_status "Tagging images for registry: $REGISTRY"
    docker tag "${IMAGE_NAME}:${TAG}" "${REGISTRY}/${IMAGE_NAME}:${TAG}"
    docker tag "${IMAGE_NAME}-sagemaker:${TAG}" "${REGISTRY}/${IMAGE_NAME}-sagemaker:${TAG}"
fi

# Display built images
print_status "Built images:"
docker images | grep "${IMAGE_NAME}"

print_status "Build completed successfully!"
print_status "To run the application: docker run -p 8501:8501 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 ${IMAGE_NAME}:${TAG}"
print_status "The Ollama service and selected local model must be running on the host."