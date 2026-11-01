# CLAUDE.md - Medical Assistant Bot

This document provides comprehensive guidance for Claude instances working with the medical-assistant-bot codebase.

## 🏥 Project Overview

The **Medical Assistant Bot** is an agentic AI system that provides conversational medical symptom analysis using:
- **Local inference**: Text analysis via Ollama; optional local vision model for images
- **RAG (Retrieval-Augmented Generation)**: FAISS vector search with medical knowledge bases
- **LangChain agent framework**: Tool orchestration and conversation memory
- **Streamlit UI**: Web interface for user interactions

**Key Design Philosophy**: Constrains LLM outputs to only verified medical data, reducing hallucinations through contextual grounding.

## 📁 Project Structure

```
medical-assistant-bot/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # User documentation
├── src/                     # Core application modules
│   ├── medical_agent_langchain.py  # LangChain agent implementation
│   ├── medical_tools.py     # Medical analysis tools
│   ├── vision_tools.py      # Optional local image analysis
│   └── ollama_runtime.py    # Local Ollama availability and resource checks
├── data/                    # Medical knowledge base (CSV files)
│   ├── disease_symptoms.csv
│   ├── disease_symptom_severity.csv
│   ├── disease_precautions.csv
│   └── disease_symptom_description.csv
├── indices/                 # FAISS vector indices
│   ├── faiss_symptom_index_medibot/
│   └── faiss_severity_index_medibot/
├── scripts/                 # Utility scripts
│   └── create_indices.py    # FAISS index generation
├── notebooks/               # Development notebooks
│   ├── data-etl.ipynb      # Data processing workflow
│   └── testbook.ipynb      # Testing and experiments
├── docs/                    # Architecture documentation
│   ├── architecture.md     # System architecture overview
│   ├── data-etl-analysis.md # Data processing analysis
│   └── market_analysis.md  # Market research
├── images/                  # Screenshots and diagrams
└── .github/workflows/       # CI/CD configuration
    └── sync-to-hf.yml      # Hugging Face deployment
```

## 🔧 Technology Stack

### Core Dependencies
- **Frontend**: Streamlit (`streamlit`)
- **LLM Framework**: LangChain (`langchain`, `langchain-community`, `langchain-ollama`, `langchain-huggingface`)
- **AI Models**: Local Ollama `llama3.2:1b` for text; optional installed Ollama vision model
- **Vector Search**: FAISS (`faiss-cpu`) with Sentence Transformers (`sentence-transformers`)
- **Data Processing**: Pandas (`pandas`), NumPy (`numpy`)
- **Image Handling**: Pillow (`Pillow`), OpenCV (`opencv-python-headless`)

### Model Configuration
- **Primary LLM**: Local Ollama `llama3.2:1b` (configurable with `OLLAMA_MODEL`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Indices**: Two FAISS indices (symptom-disease mapping, severity scoring)

## 🏗️ Architecture Components

### 1. Agent System (`src/medical_agent_langchain.py`)
- **AgenticMedicalAssistant**: Main LangChain agent coordinator
- **Tool Orchestration**: Manages medical analysis workflow
- **Conversation Memory**: Maintains context across interactions
- **Error Handling**: Robust error recovery and user feedback

### 2. Medical Tools (`src/medical_tools.py`)
- **analyze_symptoms_direct()**: Core symptom-to-disease matching via FAISS
- **analyze_combined_symptoms()**: Combines text + visual symptoms
- **analyze_symptom_severity()**: Severity assessment using FAISS
- **get_disease_description()**: Retrieves verified disease information
- **get_disease_precautions()**: Provides preventive measures

### 3. Vision Tools (`src/vision_tools.py`)
- **analyze_medical_image()**: Optional local Ollama vision integration; disabled until a local model and available memory are verified
- **Medical term translation**: Converts technical terms to common language
- **Image validation**: Comprehensive format and size checking
- **Session state integration**: Accesses Streamlit session data

### 4. Data Pipeline
- **Raw Data**: `/data/original/` contains source CSV files
- **Processed Data**: `/data/` contains cleaned, normalized CSV files
- **Vector Indices**: `/indices/` contains FAISS embeddings for fast similarity search

## 🚀 Development Workflow

### Local Development Setup
```bash
# Clone and setup
git clone <repository-url>
cd medical-assistant-bot
pip install -r requirements.txt

# Install and start Ollama, then pull the local text model
ollama pull llama3.2:1b

# Generate FAISS indices (if not present)
python scripts/create_indices.py

# Run application
python -m streamlit run app.py
```

### Key Configuration Points
1. **Local Model Configuration**: Ollama endpoint and model via `OLLAMA_BASE_URL` and `OLLAMA_MODEL`
2. **Resource Loading**: FAISS indices and CSV files loaded via `@st.cache_resource`
3. **Session Management**: Image data stored in Streamlit session state
4. **Memory Management**: Automatic conversation trimming to prevent token limits

## 📊 Data Architecture

### Data Sources
- **disease_symptoms.csv**: Disease → symptoms mappings (4,920 entries)
- **disease_symptom_severity.csv**: Symptom severity scores (1-7 scale, 133 symptoms)
- **disease_precautions.csv**: Disease → precautionary measures (41 diseases)
- **disease_symptom_description.csv**: Detailed disease descriptions (41 diseases)

### ETL Process
1. **Text Normalization**: Lowercase, whitespace cleaning, underscore removal
2. **Column Consolidation**: Multiple symptom/precaution columns → single comma-separated fields
3. **Deduplication**: Remove duplicate disease-symptom combinations
4. **FAISS Indexing**: Generate vector embeddings for fuzzy matching

### Vector Search Strategy
- **Symptom Index**: Maps user input to disease matches with confidence scores
- **Severity Index**: Enables symptom severity assessment
- **Similarity Threshold**: 1.5 distance threshold for relevant matches
- **Fuzzy Matching**: Handles variations in symptom descriptions

## 🔀 User Interaction Flow

1. **Input Collection**: Text symptoms and/or image upload
2. **Agent Orchestration**: LangChain agent determines tool sequence
3. **Tool Execution**: 
   - Image analysis (if present) → extract visual symptoms
   - Symptom analysis → FAISS search for disease matches
   - Description retrieval → get verified disease information
4. **Response Generation**: Formatted output with confidence scores
5. **Follow-up Support**: Contextual conversation with memory

## 🧪 Testing and Quality Assurance

### Development Tools
- **Jupyter Notebooks**: `/notebooks/data-etl.ipynb` for data exploration
- **Testing Framework**: pytest for unit testing
- **Validation Scripts**: FAISS index verification in `create_indices.py`

### Deployment Pipeline
- **GitHub Actions**: Automatic sync to Hugging Face Spaces on main branch push
- **Hugging Face Integration**: Live demo at `https://huggingface.co/spaces/bott-wa/medical-assistant-bot`
- **Environment Management**: Streamlit secrets for API key management

## ⚠️ Important Constraints and Guidelines

### Medical Disclaimer Requirements
- **Always include disclaimers**: This is NOT a replacement for professional medical advice
- **Emergency guidance**: Direct users to emergency services for urgent situations
- **Professional consultation**: Recommend healthcare provider consultation

### Technical Limitations
- **Local Vision Model**: 1MB PNG/JPEG upload limit; image analysis needs an installed local model and at least 4 GiB free RAM
- **Token Management**: Conversation auto-trimming at 100K estimated tokens
- **Local Runtime**: Handle Ollama availability, missing models, memory pressure, and timeouts
- **Memory Usage**: FAISS indices require sufficient RAM for loading

### Data Integrity
- **Source Control**: Medical data should only come from verified CSV sources
- **No Hallucination**: Never generate medical advice outside of database content
- **Exact Descriptions**: Use get_disease_description() output verbatim

## 🔧 Common Development Tasks

### Adding New Medical Data
1. Update CSV files in `/data/` directory
2. Re-run `scripts/create_indices.py` to regenerate FAISS indices
3. Test with sample queries to verify integration

### Modifying Analysis Tools
1. Edit functions in `src/medical_tools.py`
2. Update agent prompts in `src/medical_agent_langchain.py` if needed
3. Test tool integration via agent executor

### UI/UX Changes
1. Modify `app.py` for interface changes
2. Update CSS styling in Streamlit markdown sections
3. Test responsive behavior across devices

### Model Updates
1. Update model names in `app.py` and `medical_agent_langchain.py`
2. Test compatibility with new model capabilities
3. Update fallback model chain if needed

## 🐛 Troubleshooting Guide

### Common Issues
- **FAISS Index Loading**: Ensure `allow_dangerous_deserialization=True`
- **Image Processing**: Check base64 encoding and format validation
- **Memory Errors**: Reduce conversation history or batch sizes
- **API Errors**: Implement proper retry logic and error handling

### Debugging Tools
- **Verbose Logging**: Agent executor has verbose=True for tool tracing
- **Session State**: Monitor Streamlit session state for data persistence
- **Console Output**: Print statements throughout tool execution for debugging

## 📝 Code Style and Standards

### Python Conventions
- **Type Hints**: Use typing module for function signatures
- **Docstrings**: Document all public functions and classes
- **Error Handling**: Comprehensive try-catch blocks with user-friendly messages
- **Resource Management**: Proper cleanup of file handles and API connections

### LangChain Best Practices
- **Tool Design**: Single responsibility principle for each tool
- **Agent Prompts**: Clear, specific instructions for tool usage
- **Memory Management**: Appropriate memory buffer sizes
- **Error Recovery**: Graceful fallbacks for tool failures

## 🚀 Deployment Considerations

### Hugging Face Spaces Deployment
- **File Structure**: Ensure all required files are in repository root
- **Inference**: Requires a local Ollama service; paid cloud inference is not used
- **Resource Requirements**: Adequate RAM for FAISS indices (recommend 4GB+)
- **Startup Time**: FAISS loading can take 30-60 seconds on cold start

### Performance Optimization
- **Caching**: Use `@st.cache_resource` for expensive loading operations
- **Lazy Loading**: Load models only when needed
- **Batch Processing**: Process multiple symptoms together when possible
- **Response Streaming**: Consider streaming for long responses

## 🐳 Docker Deployment

### Docker Configuration
The project includes comprehensive Docker support with multi-stage builds:

#### Available Docker Targets
- **production**: Optimized Streamlit application (port 8501)
- **sagemaker**: SageMaker-compatible inference server (port 8080)

#### Quick Start
```bash
# Build all images
./scripts/docker/build.sh

# Run locally with Docker Compose
# Run locally with Docker Compose (Ollama must run on the host)
docker-compose up medical-assistant

# Development mode with hot reload
docker-compose --profile dev up medical-assistant-dev

# SageMaker local testing
docker-compose --profile sagemaker up sagemaker-local
```

#### Manual Docker Commands
```bash
# Build production image
docker build --target production -t medical-assistant:latest .

# Run production container
docker run -p 8501:8501 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 medical-assistant:latest

# Build SageMaker image
docker build --target sagemaker -t medical-assistant-sagemaker:latest .

# Run SageMaker container
# The legacy SageMaker target does not provide the local Ollama runtime.
```

### Docker Image Optimization
- **Multi-stage builds**: Separate build and runtime stages for smaller images
- **Non-root user**: Security hardening with dedicated app user
- **Health checks**: Built-in health monitoring endpoints
- **.dockerignore**: Optimized build context excluding unnecessary files

## ☁️ Legacy Amazon SageMaker Deployment

The SageMaker files below predate the local-Ollama migration. The current Streamlit app requires an Ollama service on the local host and does not support hosted/cloud inference. Do not use these legacy steps to run the migrated chatbot.

### SageMaker Integration
The project includes full SageMaker compatibility for enterprise deployment:

#### SageMaker Architecture
- **REST API**: Flask-based inference server with `/ping` and `/invocations` endpoints
- **Model Serving**: Containerized model with automatic scaling
- **Health Monitoring**: Built-in health checks and logging
- **Security**: IAM role-based access control

#### SageMaker Endpoints
- **`/ping`**: Health check endpoint
- **`/invocations`**: Main inference endpoint (POST)
- **`/health`**: Extended health check with model status

#### Request Format
```json
{
  "symptoms": "I have a headache, fever, and sore throat",
  "analysis_type": "basic",
  "image": "base64-encoded-image-data"  // Optional
}
```

#### Response Format
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "analysis_type": "basic",
  "input_symptoms": "headache, fever, sore throat",
  "medical_analysis": {...},
  "status": "success"
}
```

### SageMaker Deployment Process

#### Prerequisites
```bash
# Install AWS CLI and configure credentials
aws configure

# Install required Python packages
pip install boto3 sagemaker
```

#### ECR Repository Setup
```bash
# Create ECR repository
aws ecr create-repository --repository-name medical-assistant-bot --region us-east-1

# Get login token and login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and tag image
docker build --target sagemaker -t medical-assistant-sagemaker:latest .
docker tag medical-assistant-sagemaker:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/medical-assistant-bot:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/medical-assistant-bot:latest
```

#### Automated Deployment
```bash
# Deploy to SageMaker using the provided script
python scripts/docker/deploy-sagemaker.py \
    --image-uri <account-id>.dkr.ecr.us-east-1.amazonaws.com/medical-assistant-bot:latest \
    --endpoint-name medical-assistant-endpoint \
    --instance-type ml.t2.medium \
    --instance-count 1
```

#### Testing SageMaker Deployment
```bash
# Test local SageMaker container
python scripts/docker/test-sagemaker.py --endpoint-type local --url http://localhost:8080

# Test live SageMaker endpoint
python scripts/docker/test-sagemaker.py --endpoint-type sagemaker --endpoint-name medical-assistant-endpoint
```

### SageMaker Configuration Options

#### Instance Types
- **Development**: `ml.t2.medium` (2 vCPU, 4GB RAM) - Cost-effective testing
- **Production**: `ml.m5.large` (2 vCPU, 8GB RAM) - Balanced performance
- **High-traffic**: `ml.c5.xlarge` (4 vCPU, 8GB RAM) - CPU-optimized

#### Auto Scaling
SageMaker supports automatic scaling based on invocation metrics:
```python
# Configure auto scaling (example)
auto_scaling_client = boto3.client('application-autoscaling')
auto_scaling_client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId='endpoint/medical-assistant-endpoint/variant/primary',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=10
)
```

### Container Environment Variables
- **`OLLAMA_BASE_URL`**: Local Ollama endpoint
- **`SAGEMAKER_PROGRAM`**: Entry point script (default: `inference.py`)
- **`SAGEMAKER_SUBMIT_DIRECTORY`**: Code directory (default: `/opt/ml/code`)
- **`SAGEMAKER_CONTAINER_LOG_LEVEL`**: Logging level (default: `20`)
- **`SAGEMAKER_REGION`**: AWS region for SageMaker

### Cost Optimization
- **Spot Instances**: Use spot instances for non-critical workloads
- **Instance Scheduling**: Scale down during low-usage periods
- **Multi-Model Endpoints**: Share infrastructure across multiple models
- **Batch Transform**: Use batch processing for large-scale inference

This documentation should enable Claude instances to effectively understand, maintain, and extend the medical-assistant-bot codebase while adhering to its architectural principles and safety requirements.