# Spelling Correction System (Business Domain Corpus)

This project implements a probabilistic spelling correction system using a domain-specific **business news corpus** instead of a traditional dictionary. The system detects non-word errors, real-word errors, semantic inconsistencies, and grammar issues while providing context-aware correction suggestions.

The project is built using **Streamlit**, **spaCy**, and a custom-built vocabulary extracted from 5,992 BUSINESS articles (over 500,000 words) from the *News Category Dataset (Kaggle)*.

---

## 📌 Features

### ✔ Non-Word Error Detection  
Detects words not found in the business corpus (e.g., *“prouct” → "product"*).

### ✔ Real-Word Error Detection  
Checks confusion pairs (e.g., *"there" vs "their" vs "they’re"*).

### ✔ Semantic Checking  
Detects incorrect verb-object pairs (e.g., *“drink rice”*, *“eat water”*).

### ✔ Grammar Checking  
Handles subject-verb agreement (e.g., *“He go to work” → “He goes to work”*).

### ✔ Ranked Suggestions  
Uses:
- Levenshtein Edit Distance  
- Corpus word frequency  
to suggest the most likely replacement.

### ✔ Business Dictionary Panel  
Displays 4,000+ business-domain words extracted from the corpus.

### ✔ Clean UI  
JetBrains-style dark interface with animated wavy underlines for error highlights.

---

## 📁 Project Structure

pelling-Correction-System/
│
├── app.py
├── clean_business_corpus.txt
├── requirements.txt
└── README.md


---

## 🧠 How the System Works

### 1. **Corpus-Based Vocabulary**
Instead of a generic English dictionary, the system reads:


This file contains cleaned text from thousands of business news articles.  
Words are extracted and counted to form:

- A domain-specific dictionary  
- A frequency model (used for probabilistic spelling suggestions)

### 2. **Edit Distance Algorithm**
The system finds candidate words within edit distance ≤ 2 and ranks them by:

1. Lowest edit distance  
2. Highest frequency in the corpus  

### 3. **Real-Word Confusion Sets**
Handles mistakes where the word exists but is wrong in context:
- *there / their / they’re*
- *to / too / two*
- *form / from*

### 4. **Semantic Rules**
Basic verb-object rules ensure meaningful corrections:
- “eat water” → incorrect  
- “drink rice” → incorrect  

### 5. **Grammar Checking**
Uses spaCy POS tagging to detect subject-verb mismatches.

---

## 🚀 Running the Application

### 1. Install dependencies
Run:

### 2. Install spaCy English model (required)


### 3. Start the Streamlit app


### 4. Ensure the corpus file exists
Make sure `clean_business_corpus.txt` is in the same folder as `app.py`.

---

## 📚 Dataset Information

The corpus used in this project is derived from:

**News Category Dataset (Kaggle)**  
Only the **BUSINESS** category was extracted (~5,992 articles).  

These were combined, cleaned, and saved into:


The final corpus contains **over 500,000 words**, meeting the assignment requirement of a minimum 100,000-word domain corpus.

---

## 🧑‍🏫 Academic Requirements (APU)

This project fulfills the following NLP assignment components:

### ✔ Candidate Techniques  
- Edit Distance  
- Bigram/Context Principles  
- Part-of-Speech tagging  
- Corpus-driven modeling  

### ✔ Design & Formulation  
- GUI-based spell-checking system  
- Domain-specific dictionary  
- Real-word and semantic error detection  

### ✔ Implementation  
- Clean and efficient Python code  
- Custom probabilistic model  
- Streamlit interface  

### ✔ Results  
- Screenshots in the report  
- Demonstration through working GUI  

---

## 🙌 Author
**Min Thant Wai**  
Asia Pacific University – Natural Language Processing  
Spelling Correction System Project (2025)

