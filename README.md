---
title: Medora AI: An Intelligent Healthcare Chatbot with Symptom Analysis
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.29.0
app_file: app.py
pinned: false
---

# Agentic AI : Medora AI: An Intelligent Healthcare Chatbot with Symptom Analysis 🏥!

A conversational AI medical assistant that can analyze symptoms, identify possible conditions, and provide health information using local Ollama inference.

## 🧠 Design Philosophy

While today's LLMs are capable of impersonating medical professionals on their own, this **agentic multimodal system** demonstrates a more disciplined, secure, and scalable design:

> It constrains the model’s outputs to **only** the information it has been explicitly given — both in **content** *and* **context**.

This architecture enables:

- Domain-specific tuning for medical scenarios  
- Reduced hallucinations through contextual grounding  
- Intelligent interaction with both **textual symptoms** and **medical images**

The result? A smarter, safer, and more controllable diagnostic assistant — **purpose-built for trust**.



## 🔗 Key Resources

- [Market Analysis / Requirements Gathering for "Medical Chatbots"](https://huggingface.co/spaces/bott-wa/medical-assistant-bot/blob/main/docs/market_analysis.md)
- [Data Extract, Transform, Load](https://huggingface.co/spaces/bott-wa/medical-assistant-bot/blob/main/docs/data-etl-analysis.md)
- [High-Level Architecture](https://huggingface.co/spaces/bott-wa/medical-assistant-bot/blob/main/docs/architecture.md)
- [Live Demo](https://huggingface.co/spaces/bott-wa/medical-assistant-bot)

## Features

- **Symptom Analysis**: Uses a local Ollama model and the included medical knowledge base
- **Image Uploads**: PNG/JPEG uploads stay local; image analysis is optional and requires a compatible installed Ollama vision model
- **Disease Matching**: Uses FAISS vector search to find matching conditions
- **Severity Assessment**: Evaluates the severity of reported symptoms
- **Medical Information**: Provides descriptions and precautions for conditions
- **Conversational Interface**: Maintains context throughout the conversation

## How to Use

1. **Describe your symptoms** in the chat interface or upload an image
2. **Ask follow-up questions** about conditions, precautions, or severity
3. **Clear conversation** when starting a new consultation

## Technology Stack

- **Frontend**: Streamlit
- **LLM**: Ollama `llama3.2:1b` by default
- **Framework**: LangChain
- **Vector Search**: FAISS
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Image Support**: Optional local Ollama vision model; not enabled by default

## Data Sources

The medical knowledge base used here is small but usable includes:
- Disease-symptom relationships
- Symptom severity scores
- Disease descriptions
- Precautionary measures

## Privacy

- Chat and optional image inference run against the local Ollama service
- Images are not sent to a paid cloud API
- Conversations are not permanently stored
- No personal health information is collected

## Local Development

### Option 1: Direct Python Setup

```bash
# Clone the repository
git clone https://github.com/wbott/medical-assistant-bot
cd medical-assistant-bot

# Install Ollama from https://ollama.com/download and make sure its service is running.
# In a separate PowerShell window, start the service only if Ollama is not already running:
# ollama serve
ollama pull llama3.2:1b

# Install Python dependencies
python -m pip install -r requirements.txt

# If the FAISS index files are unavailable or Git LFS pointers, regenerate them:
python scripts/create_indices.py

# Run the app
python -m streamlit run app.py
```

The app checks that Ollama is reachable and that `OLLAMA_MODEL` is installed. No OpenAI API key or paid API is needed. By default, text generation uses the installed model `llama3.2:1b` through `http://localhost:11434`.

```powershell
$env:OLLAMA_MODEL = "llama3.2:1b"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
python -m streamlit run app.py
```

If Ollama cannot connect or cannot load the model, the app displays a local runtime error with guidance. Confirm Ollama is running and `ollama list` includes `llama3.2:1b`.

Image upload and preview remain available. Image analysis stays disabled unless `OLLAMA_VISION_MODEL` names an installed compatible local vision model and it has been verified separately. Check that model's memory requirements before enabling it (the app requires at least 4 GiB of available RAM). Without a configured vision model, the app will not send the image to a cloud API or claim to have analyzed it.

### Option 2: Docker Deployment

For containerized deployment, run Ollama on the host. Docker Desktop can reach it at `host.docker.internal`:

```bash
# The default model is llama3.2:1b; run Ollama on the host
docker-compose up medical-assistant
```

**Development mode with hot reload:**
```bash
docker-compose --profile dev up medical-assistant-dev
```

**Access the application:**
- Streamlit UI: http://localhost:8501
- Development mode: http://localhost:8502

## Enterprise Deployment

The repository retains legacy SageMaker deployment files, but the migrated chatbot requires Ollama running locally and does not support hosted inference. No paid cloud service is used or required by this application.

## 🧪 Testing Framework

The project includes a comprehensive testing suite with three different testing frameworks to ensure reliability and quality across all components.

### Testing Frameworks Overview

#### 1. **Pytest (Python Unit Tests)** ✅
Comprehensive Python unit and integration testing for core functionality.

**Coverage includes:**
- Core medical analysis functions (`src/medical_tools.py`)
- Image processing capabilities (`src/vision_tools.py`)
- Agent functionality (`src/medical_agent_langchain.py`)
- JSON serialization and data validation
- Error handling and edge cases
- Parametrized tests for multiple scenarios

**Test files:**
- `tests/test_medical_tools.py` - Medical analysis functions
- `tests/test_vision_tools.py` - Vision and image processing
- `tests/test_medical_agent.py` - Agent and chat functionality

#### 2. **Robot Framework (End-to-End Testing)** 🤖
Keyword-driven testing for complete user workflows and system integration.

**Coverage includes:**
- System health checks and file validation
- Complete symptom analysis workflows
- Image upload and processing validation
- Conversation memory and context testing
- Error handling and recovery scenarios
- Performance benchmarking
- Medical disclaimer verification

**Test files:**
- `tests/robot/medical_assistant.robot` - Main test suite
- `tests/robot_keywords/MedicalAssistantKeywords.py` - Custom keywords

#### 3. **JUnit (Java Integration Testing)** ☕
Cross-language integration testing for system-level validation.

**Coverage includes:**
- Python-Java process execution testing
- HTTP API interaction validation
- Performance and memory usage monitoring
- Cross-platform compatibility testing
- System integration scenarios

**Test files:**
- `tests/java/MedicalAssistantIntegrationTest.java` - Integration tests
- `pom.xml` - Maven configuration

### Running Tests

#### Prerequisites
```bash
# Install testing dependencies
pip install -r requirements.txt

# The following testing packages are included:
# - pytest>=7.0.0
# - pytest-html>=3.1.0
# - pytest-cov>=4.0.0
# - pytest-mock>=3.10.0
# - robotframework>=6.0.0
# - robotframework-requests>=0.9.0
```

#### Individual Framework Commands

**Run Pytest (Python Unit Tests):**
```bash
# Basic test run
pytest tests/

# With verbose output and coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_medical_tools.py -v

# Run with HTML report
pytest tests/ --html=test-results/pytest-report.html --self-contained-html
```

**Run Robot Framework (End-to-End Tests):**
```bash
# Basic test run
robot tests/robot/medical_assistant.robot

# With custom output directory
robot --outputdir test-results/robot tests/robot/medical_assistant.robot

# Run specific test tags
robot --include smoke tests/robot/medical_assistant.robot

# Dry run (syntax check only)
robot --dryrun tests/robot/medical_assistant.robot
```

**Run JUnit Tests (requires Java 11+ and Maven):**
```bash
# Install Java and Maven first
sudo apt update
sudo apt install openjdk-11-jdk maven

# Run all tests
mvn test

# Run with specific profile
mvn test -P integration

# Fast tests only (skip slow tests)
mvn test -P fast
```

#### Comprehensive Test Runner

**Run all testing frameworks at once:**
```bash
# Execute complete test suite with reporting
python test_runner.py
```

This will:
- Run all pytest tests with coverage reporting
- Execute Robot Framework syntax validation
- Check Java/Maven environment availability
- Generate a comprehensive test report in `test-results/comprehensive-test-report.md`

#### Test Configuration Files

- **`pytest.ini`** - Pytest configuration with coverage settings
- **`robot.yaml`** - Robot Framework configuration
- **`pom.xml`** - Maven configuration for Java tests

#### Test Results and Reporting

All test results are saved to the `test-results/` directory:

```
test-results/
├── pytest-report.html          # Pytest HTML report
├── coverage-html/               # Coverage report
├── robot/                       # Robot Framework reports
│   ├── report.html
│   ├── log.html
│   └── output.xml
├── junit/                       # JUnit test reports
└── comprehensive-test-report.md # Combined report
```

#### Continuous Integration

The testing framework is designed for CI/CD integration:

```bash
# CI-friendly command (non-interactive)
pytest tests/ --tb=short --junit-xml=test-results/junit.xml
robot --outputdir test-results/robot --exitonfailure tests/robot/
```

#### Test Development Guidelines

When adding new features:

1. **Write unit tests first** (pytest) for individual functions
2. **Add integration tests** (Robot Framework) for user workflows  
3. **Include error handling tests** for edge cases
4. **Update test documentation** in this section
5. **Run full test suite** before committing changes

**Mock Testing:**
- Tests include comprehensive mocking for external dependencies
- API keys not required for most unit tests
- Offline testing capabilities for development environments

For detailed testing architecture and guidelines, see [CLAUDE.md](CLAUDE.md).

## Important Disclaimers

⚠️ **This is not a replacement for professional medical advice**
- Always consult with qualified healthcare providers for proper diagnosis
- This tool provides general information only
- Do not make medical decisions based solely on this tool's output
- In case of emergency, contact emergency services immediately
