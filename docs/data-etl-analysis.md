# 📊 Data ETL & Analysis for Medical Assistant Bot

## 🧾 Summary

This document outlines the data extraction, transformation, and loading (ETL) process used to prepare the [wbott's Medical Assistant Bot](https://github.com/wbott/medical-assistant-bot) =) 

The original data proved surprisingly robust with minimal need for transformation beyond format adjustments. Key analytical tasks like collinearity analysis, data grouping, and joins were considered but ultimately unnecessary due to the power of embedding and indexing with [FAISS](https://github.com/facebookresearch/faiss)
.

### ✅ Key Points
- **Data integrity** was strong—minor text normalization was sufficient.
- **Collinearity analysis**, clustering, or advanced joins were not required.
- **Embedding and FAISS** provided robust fuzzy matching that made elaborate data reengineering redundant.
- **Two FAISS indexes** were created: one for symptom-disease similarity and one for symptom severity scoring.

---

## 📁 Data Sources

Original datasets:
- [dataset.csv](../data/original/dataset.csv)
- [symptom_precaution.csv](../data/original/symptom_precaution.csv)
- [symptom_Description.csv](../data/original/symptom_Description.csv)
- [Symptom-severity.csv](../data/original/Symptom-severity.csv)

Cleaned datasets:
- [disease_symptoms.csv](../data/original/disease_symptoms.csv)
- [disease_precautions.csv](../data/original/disease_precautions.csv)
- [disease_symptom_description.csv](../data/original/disease_symptom_description.csv)
- [disease_symptom_severity.csv](../data/original/disease_symptom_severity.csv)


These CSV files were structured with disease or symptom identifiers and corresponding values (symptoms, descriptions, precautions, etc.), sometimes spread across multiple columns.

---

## 🧼 Data Cleaning & Transformation

### 1. 🧹 Basic Text Cleanup
- **Underscores** were replaced with spaces.
- **Extraneous whitespace** was stripped.
- **All text was lowercased** to avoid case-sensitive mismatches during search.

This significantly improved the performance of downstream FAISS vector matching.

---

### 2. 🩺 Symptom Data Reformatting
- The `disease_symptoms.csv` file had up to **17 symptom columns**, most of them null.
- These were **collapsed into a single comma-separated `symptoms` column**.
- This normalized format simplified both vector indexing and user input parsing.

---

### 3. 📜 Description & Deduplication
- **Symptom and disease descriptions** were largely usable out of the box.
- Some **duplicate records** with identical diseases or symptoms were removed.
- Ensured consistency in phrasing and formatting to aid lookup accuracy.

---

### 4. ⚠️ Precautions Reshaping
- Similar to symptoms, precautions were listed across **multiple columns**.
- All were **collapsed into a single `precautions` column**, allowing for compact storage and indexing.

---

## 🧠 Embedding & FAISS Integration

The real power came from **FAISS-based similarity indexing**, enabling intelligent fuzzy matching across:

- **Symptom-to-disease** relationships
- **Symptom-to-severity** scoring

### Created Indexes:
- `faiss_symptom_index_medibot`
- `faiss_severity_index_medibot`

These enabled:
- Recognition of variations like `"runny nose"` vs `"running nose"`.
- Severity scoring even with loosely matched or rephrased inputs.
- Simplified pipeline without needing to reengineer core datasets.

---

## ✅ Conclusion

Despite initial expectations, the dataset required **minimal restructuring**. Once formatted for consistency, the **FAISS-powered similarity search** eliminated the need for deeper statistical preprocessing. This let us keep our focus on agent orchestration and conversational design rather than exhaustive data wrangling.

The result: a lightweight yet powerful data backbone for [wbott's Medical Assistant Bot](https://github.com/wbott/medical-assistant-bot), deployed on [wbott's Hugging Face Spaces](https://huggingface.co/spaces/bott-wa/medical-assistant-bot).

