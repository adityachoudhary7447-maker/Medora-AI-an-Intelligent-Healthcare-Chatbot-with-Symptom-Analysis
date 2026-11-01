"""
Create FAISS Indices for Medical Assistant
=========================================

This script creates the FAISS indices from your CSV files.
Run this locally before deploying to Hugging Face.
"""

import os
import sys
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def create_faiss_indices(data_path="./data", output_path="./indices"):
    """Create FAISS indices from CSV files."""
    
    print("🚀 Starting FAISS index creation...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load embedding model
    print("📥 Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'device': 'cpu', 'batch_size': 32}
    )
    
    # Load CSV files
    print("📊 Loading CSV files...")
    df_disease_symptoms = pd.read_csv(f"{data_path}/disease_symptoms.csv")
    df_disease_symptom_severity = pd.read_csv(f"{data_path}/disease_symptom_severity.csv")
    
    # Create documents for symptom index
    print("📝 Creating symptom documents...")
    symptom_documents = []
    
    # Process disease_symptoms.csv
    for _, row in df_disease_symptoms.iterrows():
        disease = row['disease']
        symptoms = row['symptoms']
        
        # Create a document for each disease-symptoms pair
        doc = Document(
            page_content=symptoms,
            metadata={"disease": disease}
        )
        symptom_documents.append(doc)
    
    # Create FAISS index for symptoms
    print("🔨 Creating symptom FAISS index...")
    faiss_symptom_index = FAISS.from_documents(
        symptom_documents, 
        embedding_model
    )
    
    # Save symptom index
    print("💾 Saving symptom index...")
    faiss_symptom_index.save_local(f"{output_path}/faiss_symptom_index_medibot")
    
    # Create documents for severity index
    print("📝 Creating severity documents...")
    severity_documents = []
    
    # Get unique symptoms from severity data
    unique_symptoms = df_disease_symptom_severity['symptom'].unique()
    
    for symptom in unique_symptoms:
        doc = Document(
            page_content=symptom,
            metadata={"symptom": symptom}
        )
        severity_documents.append(doc)
    
    # Create FAISS index for severity
    print("🔨 Creating severity FAISS index...")
    faiss_severity_index = FAISS.from_documents(
        severity_documents,
        embedding_model
    )
    
    # Save severity index
    print("💾 Saving severity index...")
    faiss_severity_index.save_local(f"{output_path}/faiss_severity_index_medibot")
    
    print("✅ FAISS indices created successfully!")
    print(f"📁 Indices saved in: {output_path}")
    
    # Verify indices
    print("\n🔍 Verifying indices...")
    
    # Test loading
    try:
        test_symptom = FAISS.load_local(
            f"{output_path}/faiss_symptom_index_medibot",
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        test_severity = FAISS.load_local(
            f"{output_path}/faiss_severity_index_medibot",
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        
        print(f"✅ Symptom index: {len(symptom_documents)} documents")
        print(f"✅ Severity index: {len(severity_documents)} documents")
        
        # Test search
        test_result = test_symptom.similarity_search("fever headache", k=3)
        print(f"\n🧪 Test search for 'fever headache':")
        for i, doc in enumerate(test_result):
            print(f"  {i+1}. {doc.metadata['disease']}: {doc.page_content[:50]}...")
            
    except Exception as e:
        print(f"❌ Error verifying indices: {e}")
        return False
    
    return True


def prepare_csv_files(input_path, output_path="./data"):
    """
    Prepare CSV files in the correct format.
    Adjust this function based on your actual CSV structure.
    """
    
    print("📋 Preparing CSV files...")
    os.makedirs(output_path, exist_ok=True)
    
    # Copy CSV files to data directory
    import shutil
    
    files_to_copy = [
        "disease_symptoms.csv",
        "disease_symptom_severity.csv",
        "disease_precautions.csv",
        "disease_symptom_description.csv"
    ]
    
    for file in files_to_copy:
        src = os.path.join(input_path, file)
        dst = os.path.join(output_path, file)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"✅ Copied {file}")
        else:
            print(f"⚠️ Warning: {file} not found in {input_path}")
    
    # If your CSV has different structure, transform it here
    # Example: If symptoms are in separate columns
    """
    df = pd.read_csv(f"{input_path}/your_original_file.csv")
    
    # Transform to required format
    disease_symptoms = []
    for _, row in df.iterrows():
        symptoms = []
        for col in df.columns:
            if col.startswith('symptom_') and pd.notna(row[col]):
                symptoms.append(row[col])
        
        if symptoms:
            disease_symptoms.append({
                'disease': row['disease'],
                'symptoms': ', '.join(symptoms)
            })
    
    # Save transformed data
    pd.DataFrame(disease_symptoms).to_csv(
        f"{output_path}/disease_symptoms.csv", 
        index=False
    )
    """


if __name__ == "__main__":
    # Set your paths
    INPUT_DATA_PATH = "./"  # Your original data
    OUTPUT_DATA_PATH = "./data"  # Where to save processed CSVs
    OUTPUT_INDICES_PATH = "./indices"  # Where to save FAISS indices
    
    # Step 1: Prepare CSV files
    prepare_csv_files(INPUT_DATA_PATH, OUTPUT_DATA_PATH)
    
    # Step 2: Create FAISS indices
    success = create_faiss_indices(OUTPUT_DATA_PATH, OUTPUT_INDICES_PATH)
    
    if success:
        print("\n🎉 All done! Your indices are ready for deployment.")
        print("\n📦 Next steps:")
        print("1. Upload the 'data/' and 'indices/' folders to your HF Space")
        print("2. Upload app.py, requirements.txt, and other Python files")
        print("3. Configure the local Ollama model and endpoint")
        print("4. Your medical assistant will be ready to use!")
    else:
        print("\n❌ There were errors creating the indices. Please check the logs above.")