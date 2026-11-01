# Agentic RAG: Medical Assistant Bot / Symptom Diagnosis

## Overview

This document outlines the architecture of an intelligent medical assistant bot designed for symptom diagnosis using Retrieval-Augmented Generation (RAG) technology. The system leverages multiple data sources and an agentic approach to provide accurate diagnostic suggestions and medical information based on user-reported symptoms.

![Architecture Overview](../images/architecture_overview.png)

## System Architecture Components

### Step 1: User Input Collection
The process begins with the **most critical step** where users input a series of symptoms they are experiencing. This initial query serves as the foundation for the diagnostic process, allowing users to describe their symptoms in natural language to identify potential medical conditions, causes, or diseases.

### Step 2: Retrieval Agent Configuration
The system employs a pre-configured **Retrieval Agent** that contains:
- **System Prompt**: Core instructions and configuration settings
- **User Query Integration/User Input Prompt**: Human-provided symptom description
- **Chat Memory**: Maintains context from ongoing conversations for continuous dialogue
- **LLM Integration**: Partially utilizes the LLM to parse and process user input as needed

The system prompt includes comprehensive instructions that enable the Language Model to invoke various functions and access different data stores as needed.

### Step 3: Data Store Access
The retrieval agent connects to multiple specialized data repositories:
- **Image Symptom Extraction**: Optional local Ollama vision model; image analysis remains disabled unless an installed model and sufficient memory are confirmed
- **Disease Symptoms Database**: Vector-based FAISS index containing symptom-disease mappings
- **Symptom Severity Database**: Vector-based FAISS index with severity classifications
- **Disease Descriptions**: Comma-separated value files containing detailed disease information
- **Disease Precautions**: CSV files with preventive measures and care instructions


### Step 4: Information Sufficiency Assessment
The Language Model evaluates whether the retrieved information from Step 3 is sufficient to generate a comprehensive response. This decision point determines the next course of action in the diagnostic process.

### Step 5: Additional Information Retrieval
If the initial data retrieval is insufficient (for example, when users request both symptoms and specific precautions), the LM invokes additional function calls to gather more information from the data stores. This may include:
- Extended symptom severity descriptions
- Detailed precautionary measures
- Additional contextual medical information

### Step 6: Response Generation
Once all necessary information has been collected, the Language Model processes the data and generates a comprehensive response. This response is formatted as an LLM-generated output that provides diagnostic suggestions, symptom analysis, and relevant medical guidance.

### Step 7: User Interaction and Continuation
The generated response is delivered to the user, who can then:
- Review the diagnostic information provided
- Ask follow-up questions to continue the conversation
- Request more specific details about identified conditions
- Start a new diagnostic session

The chat memory component ensures continuity in conversations, allowing users to build upon previous interactions. For example, if the system initially suggests malaria as a potential diagnosis, users can subsequently ask "tell me more about malaria" or "tell me more about the most probable cause" to continue the diagnostic dialogue.

## Key Features

- **Multi-Source Data Integration**: Combines vector databases and structured files for comprehensive medical knowledge
- **Multi-Agent / Multi-modal**: Combines multiple agents for text, image multi-modal analysis
- **Conversational Memory**: Maintains context across multiple interactions
- **Adaptive Information Retrieval**: Dynamically fetches additional data based on user needs
- **Iterative Diagnostic Process**: Supports ongoing dialogue for refined diagnosis and information gathering

## Usage Flow

The system supports both single-query interactions, combination queries with image and text and extended diagnostic conversations, making it suitable for initial symptom assessment as well as detailed medical information exploration. Users can engage with the system multiple times, building upon previous conversations to gain deeper insights into potential medical conditions and appropriate precautionary measures, or clear and start over.