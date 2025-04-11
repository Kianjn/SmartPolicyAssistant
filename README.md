# SmartPolicyAssistant 🚀

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-00A67E)](https://www.langchain.com/)

A GenAI-powered assistant using RAG to analyze, summarize, and compare EU National Energy & Climate Plans (NECPs).

## 📋 Overview

This capstone project demonstrates a GenAI-powered assistant designed to help users understand, analyze, and compare lengthy and complex energy policy documents, specifically focusing on the National Energy and Climate Plans (NECPs) of EU member states. It leverages Retrieval-Augmented Generation (RAG) to provide accurate answers based on the source documents.

## ✨ Features

*   **📄 Document Ingestion:** Processes raw text extracted from NECP PDF documents
*   **🔍 Vector Indexing:** Creates a searchable vector index (using FAISS) from document chunks based on semantic similarity using Google's embedding models
*   **🤖 Retrieval-Augmented Generation (RAG):** Finds relevant document sections based on user queries
*   **💡 Generative Q&A:** Uses Google Gemini models via LangChain to generate answers grounded in the retrieved context
*   **🌍 Cross-Document Comparison:** Capable of answering questions that require synthesizing information from multiple documents (e.g., comparing targets between countries)
*   **🖥️ Interactive UI:** Provides a user-friendly chat interface built with Streamlit

## 🛠️ Technology Stack

*   **Language:** Python 3.10+
*   **Core AI/ML:** 
    * LangChain
    * Google Generative AI (Gemini API)
    * Sentence Transformers (via LangChain)
*   **Vector Store:** FAISS (Facebook AI Similarity Search)
*   **Web Framework:** Streamlit
*   **Environment:** Conda
*   **Other:** python-dotenv

## 📁 Project Structure

```plaintext
SmartPolicyAssistant/
├── data/
│   ├── input/           # Directory for source PDF files
│   └── processed/       # Directory for processed text files
├── src/
│   ├── preprocessing/   # Scripts for PDF processing and summarization
│   │   ├── pdf_to_txt.py
│   │   └── summarize.py
│   ├── indexing/       # Scripts for creating and managing FAISS index
│   │   └── policy_qa.py
│   ├── core/          # Core chat functionality
│   │   ├── chat_with_policy.py
│   │   └── .env       # Stores the Google API Key 
│   └── ui/           # Streamlit User Interface
│       └── ui_chat.py
├── models/          # Directory for storing FAISS index files
│   ├── index.faiss
│   └── index.pkl
├── environment.yml  # Conda environment definition file
├── LICENSE         # Contains the MIT License text
└── README.md      # This file
```

## 🚀 Quick Start

### Prerequisites

- [Git](https://git-scm.com/)
- [Conda/Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Kianjn/SmartPolicyAssistant.git
   cd SmartPolicyAssistant
   ```

2. **Set Up Environment**
   ```bash
   # Using environment.yml
   conda env create -f environment.yml
   conda activate policy_env
   
   # OR Manual Installation
   conda create --name policy_env python=3.10 -y
   conda activate policy_env
   conda install pytorch cpuonly faiss-cpu -c pytorch -c conda-forge -y
   pip install --upgrade pip
   pip install langchain langchain-google-genai google-generativeai langchain-community sentence-transformers python-dotenv streamlit faiss-cpu
   ```

3. **Configure API Key**
   ```bash
   # Create .env file in src/core/
   echo "GOOGLE_API_KEY=your_api_key_here" > src/core/.env
   ```

4. **Prepare Data**
   - Place NECP PDFs in `data/input/`
   - Run preprocessing scripts to convert PDFs to text
   - Processed files will be stored in `data/processed/`

5. **Create FAISS Index**
   ```bash
   cd src/indexing
   python policy_qa.py
   ```

## 💬 Usage

### Web Interface
```bash
cd src/ui
streamlit run ui_chat.py
```
Access the interface at `http://localhost:8501`

### Command Line Interface
```bash
cd src/core
python chat_with_policy.py
```

### Example Questions

*   "What is Germany's target for renewable energy share in 2030?"
*   "Summarize Italy's main policies for the transport sector."
*   "Compare the greenhouse gas emission reduction targets of France and Spain for 2030."
*   "What measures does Austria mention regarding energy efficiency in buildings?"

## 🔮 Future Improvements

*   **Metadata Filtering:** Enhance retrieval by filtering based on country metadata attached to chunks
*   **Advanced RAG Techniques:** Explore Multi-Query Retriever, MMR, Parent Document Retriever, or HyDE
*   **Enhanced Comparison Logic:** Implement more sophisticated prompting for comparison tasks
*   **Source Highlighting:** Show which parts of source documents were used in answers
*   **Evaluation Framework:** Implement metrics for response quality and accuracy
*   **Integrated Summarization:** Add summarization capability directly into the workflow/UI

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

*   Data Sources: European Commission / Member State NECP documents
*   Key Libraries: LangChain, Streamlit, FAISS, Google Generative AI
