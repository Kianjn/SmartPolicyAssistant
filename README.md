# SmartPolicyAssistant 🚀

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-00A67E)](https://www.langchain.com/)

A GenAI-powered assistant using RAG to analyze, summarize, and compare EU National Energy & Climate Plans (NECPs).

## 📋 Overview

This project demonstrates a GenAI-powered assistant designed to help users understand, analyze, and compare lengthy and complex energy policy documents, specifically focusing on the National Energy and Climate Plans (NECPs) of EU member states. It leverages Retrieval-Augmented Generation (RAG) and structured data analysis to provide accurate answers based on the source documents.

## ✨ Features

*   **📄 Document Processing:** Processes and extracts structured data from NECP documents
*   **📊 Data Aggregation:** Combines and normalizes data from multiple sources into a unified format
*   **🤖 LLM-Powered Query Parsing:** Uses Google Gemini models to understand and parse natural language queries
*   **🔍 Structured Data Retrieval:** Efficiently retrieves relevant data based on parsed queries
*   **💡 Generative Q&A:** Uses Google Gemini models via LangChain to generate answers grounded in the retrieved data
*   **🌍 Cross-Document Comparison:** Capable of answering questions that require synthesizing information from multiple documents
*   **🖥️ Interactive UI:** Provides a user-friendly chat interface built with Streamlit

## 🛠️ Technology Stack

*   **Language:** Python 3.10+
*   **Core AI/ML:** 
    * LangChain
    * Google Generative AI (Gemini API)
    * Pandas for data manipulation
*   **Web Framework:** Streamlit
*   **Environment:** Conda
*   **Other:** python-dotenv

## 📁 Project Structure

```plaintext
SmartPolicyAssistant/
├── data/
│   ├── input/           # Directory for source documents
│   └── processed/       # Directory for processed and aggregated data
├── src/
│   ├── preprocessing/   # Scripts for data processing and aggregation
│   │   ├── pdf_to_txt.py
│   │   └── summarize.py
│   ├── indexing/       # Scripts for query processing and data retrieval
│   │   └── policy_qa.py
│   └── ui/            # Streamlit User Interface
│       └── ui_chat.py
├── models/            # Directory for storing model-related files
├── .env              # Environment variables (API keys)
├── LICENSE           # Contains the MIT License text
└── README.md        # This file
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
   # Create and activate conda environment
   conda create --name policy_env python=3.10 -y
   conda activate policy_env
   
   # Install required packages
   pip install --upgrade pip
   pip install langchain langchain-google-genai google-generativeai langchain-community python-dotenv streamlit pandas
   ```

3. **Configure API Key**
   ```bash
   # Create .env file in project root
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   ```

4. **Prepare Data**
   - Place source documents in `data/input/`
   - Run preprocessing scripts to process and aggregate data
   - Processed files will be stored in `data/processed/`

## 💬 Usage

### Web Interface
```bash
cd src/ui
streamlit run ui_chat.py
```
Access the interface at `http://localhost:8501`

### Command Line Interface
```bash
cd src/indexing
python policy_qa.py
```

### Example Questions

*   "What is Germany's target for renewable energy share in 2030?"
*   "Compare the greenhouse gas emission reduction targets of France and Spain for 2030."
*   "What are the energy efficiency targets for buildings in Austria?"
*   "List all countries with renewable energy targets above 40% for 2030."

## 🔮 Future Improvements

*   **Enhanced Data Processing:** Improve data extraction and normalization from source documents
*   **Advanced Query Understanding:** Implement more sophisticated query parsing and intent recognition
*   **Data Visualization:** Add interactive charts and graphs for data comparison
*   **Multi-language Support:** Enable querying in multiple languages
*   **Evaluation Framework:** Implement metrics for response quality and accuracy
*   **Real-time Updates:** Add capability to process and incorporate new data sources

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

*   Data Sources: European Commission / Member State NECP documents
*   Key Libraries: LangChain, Streamlit, Google Generative AI, Pandas
