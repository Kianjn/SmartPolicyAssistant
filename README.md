# SmartPolicyAssistant 🚀

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Demo-FF4B4B)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG%20Orchestration-00A67E)](https://www.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/AI%20Model-Google%20Gemini-4285F4)](https://ai.google.dev/)

**Unlock Insights from Complex EU Energy Policy with Generative AI**

SmartPolicyAssistant is an advanced AI-powered tool designed to tackle the challenge of analyzing, summarizing, and comparing dense European Union National Energy & Climate Plans (NECPs). Leveraging cutting-edge Retrieval-Augmented Generation (RAG) techniques with Google's Gemini models, this project provides a streamlined way to extract structured information and generate accurate, context-aware answers grounded in source documents.

---

## 🌟 Project Highlights

*   **Addresses a Real-World Problem:** Automates the laborious task of navigating and comparing complex, lengthy policy documents (EU NECPs).
*   **Advanced AI Implementation:** Demonstrates practical application of RAG architecture using LangChain and Google Gemini for reliable, source-based Q&A.
*   **Structured Data Focus:** Goes beyond simple text summarization by extracting and utilizing structured data for precise analysis and comparison across documents.
*   **Interactive & Accessible:** Features a user-friendly Streamlit interface for intuitive interaction and exploration.

## 🎥 Learn More & See it in Action

Dive deeper into the project's methodology, implementation, and results:

*   **▶️ YouTube Demo:** [Watch a walkthrough of the SmartPolicyAssistant](https://youtu.be/D3LE5F19Ahc)
*   **📝 Blog Post:** [Read the detailed project breakdown and insights on Medium](https://medium.com/@kianjafarinezhad/automating-necp-analysis-a-capstone-project-leveraging-genai-for-structured-data-extraction-and-6aae7187627f)
*   **💻 Kaggle Notebook:** [Explore the code and analysis process](https://www.kaggle.com/code/kianjn/smart-policy-assistant)

## 📋 Core Functionality

*   **📄 Intelligent Document Processing:** Ingests NECP documents (PDFs initially), extracts text, and prepares data for analysis.
*   **📊 Structured Data Extraction & Aggregation:** Identifies key metrics and policy points, normalizing them into a unified, queryable format using Pandas.
*   **🤖 Sophisticated Query Understanding:** Employs Google Gemini models to interpret natural language questions about energy policies.
*   **🔍 RAG-Powered Retrieval:** Efficiently locates relevant information chunks from the processed data based on the user's query.
*   **💡 Contextual Answer Generation:** Leverages Google Gemini via LangChain to synthesize retrieved information and generate accurate, natural language answers, citing sources.
*   **🌍 Cross-Document Analysis & Comparison:** Enables users to ask questions requiring data synthesis across multiple NECPs (e.g., comparing targets between countries).
*   **🖥️ Interactive Chat Interface:** Provides an intuitive Streamlit-based web application for easy querying and result visualization.

## 🛠️ Technology Stack

*   **Core Language:** Python (3.10+)
*   **AI/ML Frameworks:**
    *   LangChain: Orchestration of the RAG pipeline, prompt management, and LLM interaction.
    *   Google Generative AI: Powering the core LLM capabilities (Gemini API) for understanding and generation.
*   **Data Handling:** Pandas for efficient data manipulation, structuring, and aggregation.
*   **Web Interface:** Streamlit for rapid development of the interactive user interface.
*   **Environment Management:** Conda
*   **Utilities:** python-dotenv for secure API key management.

## 📁 Project Structure

```plaintext
SmartPolicyAssistant/
├── data/
│   ├── input/           # Raw source documents (e.g., NECP PDFs)
│   └── processed/       # Intermediate files, extracted text, structured data outputs
├── src/
│   ├── preprocessing/   # Scripts for PDF conversion, text cleaning, summarization (if applicable)
│   │   ├── pdf_to_txt.py # Example script
│   │   └── summarize.py  # Example script
│   ├── indexing/        # Core logic for RAG: query parsing, data retrieval, answer generation
│   │   └── policy_qa.py # Main Q&A logic, potentially CLI entry point
│   └── ui/              # Streamlit application code
│       └── ui_chat.py   # Main UI script
├── models/              # Potentially for storing fine-tuned models or embeddings (if used)
├── .env                 # Secure storage for API keys (!!! ADD TO .gitignore !!!)
├── requirements.txt     # List of Python dependencies (Consider adding this)
├── LICENSE              # Project License (MIT)
└── README.md            # You are here!

*(**Note:** Ensure your `.env` file is included in your `.gitignore` to prevent accidental key exposure! Consider adding a `requirements.txt` for pip users.)*

## 🚀 Getting Started

### Prerequisites

*   [Git](https://git-scm.com/) installed.
*   [Conda (or Miniconda)](https://docs.conda.io/en/latest/miniconda.html) installed.
*   Access to Google Generative AI (Gemini API) and an API Key.

### Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Kianjn/SmartPolicyAssistant.git
    cd SmartPolicyAssistant
    ```

2.  **Create & Activate Conda Environment:**
    ```bash
    conda create --name policy_env python=3.10 -y
    conda activate policy_env
    ```

3.  **Install Dependencies:**
    *(Consider creating a `requirements.txt` file (`pip freeze > requirements.txt`) and using `pip install -r requirements.txt`)*
    ```bash
    pip install --upgrade pip
    pip install langchain langchain-google-genai google-generativeai langchain-community python-dotenv streamlit pandas PyPDF2 # Added PyPDF2 assuming PDF processing
    # Add any other specific dependencies here
    ```

4.  **Configure Google API Key:**
    Create a `.env` file in the project's root directory:
    ```bash
    echo "GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY_HERE" > .env
    ```
    Replace `YOUR_GOOGLE_API_KEY_HERE` with your actual key.

5.  **Prepare Data:**
    *   Place your source NECP documents (e.g., PDFs) into the `data/input/` directory.
    *   Run the necessary preprocessing scripts located in `src/preprocessing/` to convert, clean, and potentially structure the data. (You might need to add specific instructions here depending on your scripts).
    *   Processed data required for the Q&A system should reside in `data/processed/`.

## 💬 Usage

### Option 1: Interactive Web Interface (Recommended)

1.  Navigate to the UI directory:
    ```bash
    cd src/ui
    ```
2.  Launch the Streamlit application:
    ```bash
    streamlit run ui_chat.py
    ```
3.  Open your web browser and go to `http://localhost:8501` (or the URL provided by Streamlit).

### Option 2: Command Line Interface (for testing/scripting)

1.  Navigate to the indexing logic directory:
    ```bash
    cd src/indexing
    ```
2.  Run the Q&A script (this might require specific arguments depending on `policy_qa.py`'s implementation):
    ```bash
    python policy_qa.py --query "Your question here"
    # Or run interactively if the script supports it
    python policy_qa.py
    ```

### Example Questions

*   "What is Germany's target for renewable energy share in gross final consumption for 2030?"
*   "Compare the planned reduction in greenhouse gas emissions for France and Spain by 2030 relative to 1990 levels."
*   "Summarize Austria's strategy for improving energy efficiency in the building sector."
*   "Which countries explicitly mention targets for offshore wind capacity expansion by 2030?"
*   "List the primary policy measures mentioned by Italy to support renewable energy deployment."

## 🔮 Future Roadmap & Potential Enhancements

*   **Enhanced Data Extraction:** Implement more robust NLP techniques (e.g., NER, relation extraction) for finer-grained structured data capture from PDFs/text.
*   **Advanced Query Decomposition:** Break down complex multi-part questions for more accurate retrieval and synthesis.
*   **Interactive Visualization:** Integrate plotting libraries (e.g., Plotly, Altair) within Streamlit to visualize comparative data.
*   **Multi-lingual Support:** Fine-tune or adapt models to handle queries and potentially documents in multiple EU languages.
*   **Scalability & Performance:** Optimize data loading, indexing (e.g., using vector databases like ChromaDB or FAISS), and API calls for larger datasets.
*   **Formal Evaluation Framework:** Develop metrics (e.g., RAGAS, BLEU, ROUGE, factual consistency checks) to quantitatively assess response quality.
*   **Real-time Data Integration:** Build pipelines to automatically fetch, process, and incorporate updated NECP documents or related policy news.
*   **User Feedback Loop:** Implement mechanisms for users to rate answer quality, helping to refine the system.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or want to contribute code, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

Please ensure your code adheres to standard Python best practices and includes relevant documentation or tests where applicable.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.

## 🙏 Acknowledgements

*   **Data Source:** National Energy and Climate Plans provided by EU Member States via the European Commission.
*   **Core Technologies:** Immense thanks to the teams behind LangChain, Streamlit, Google Generative AI, and Pandas for their powerful open-source tools and services.
*   **Inspiration:** The broader open-source AI community pushing the boundaries of natural language understanding and generation.