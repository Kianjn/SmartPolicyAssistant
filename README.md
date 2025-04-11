# SmartPolicyAssistant
GenAI-powered assistant using RAG to analyze, summarize, and compare EU National Energy &amp; Climate Plans (NECPs).

This capstone project demonstrates a GenAI-powered assistant designed to help users understand, analyze, and compare lengthy and complex energy policy documents, specifically focusing on the National Energy and Climate Plans (NECPs) of EU member states. It leverages Retrieval-Augmented Generation (RAG) to provide accurate answers based on the source documents.

## Features

*   **Document Ingestion:** Processes raw text extracted from NECP PDF documents.
*   **Vector Indexing:** Creates a searchable vector index (using FAISS) from document chunks based on semantic similarity using Google's embedding models.
*   **Retrieval-Augmented Generation (RAG):** Finds relevant document sections based on user queries.
*   **Generative Q&A:** Uses Google Gemini models via LangChain to generate answers grounded in the retrieved context.
*   **Cross-Document Comparison:** Capable of answering questions that require synthesizing information from multiple documents (e.g., comparing targets between countries).
*   **Interactive UI:** Provides a user-friendly chat interface built with Streamlit.

## Technology Stack

*   **Language:** Python 3.10+
*   **Core AI/ML:** LangChain, Google Generative AI (Gemini API), Sentence Transformers (via LangChain)
*   **Vector Store:** FAISS (Facebook AI Similarity Search)
*   **Web Framework:** Streamlit
*   **Environment:** Conda
*   **Other:** python-dotenv

## Project Structure

SmartPolicyAssistant/ # Root directory matching the repository name
├── 01.Summarization/ # (Handles initial PDF -> TXT processing - Script not included in this repo)
├── 02.RAG/ # Core RAG backend, data, and index
│ ├── FAISS/ # Default directory for the FAISS index files
│ │ ├── index.faiss
│ │ └── index.pkl
│ ├── necp_texts/ # Directory containing the processed .txt files (one per country/document)
│ │ ├── austria.txt # Example files - provide your own
│ │ └── germany.txt
│ ├── policy_qa.py # Script to create/update the FAISS index
│ ├── chat_with_policy.py# Script for command-line chat interface
│ └── .env # Stores the Google API Key (!! IMPORTANT: Add .env to .gitignore !!)
└── 03.UI/ # Streamlit User Interface
└── ui_chat.py # The Streamlit application script
├── environment.yml # (Recommended) Conda environment definition file
├── LICENSE # Contains the MIT License text
└── README.md # This file

## Setup and Installation

**1. Prerequisites:**
    *   Git: [https://git-scm.com/](https://git-scm.com/)
    *   Conda / Miniforge: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

**2. Clone Repository:**
    ```bash
    # Replace [URL of your GitHub repository] with the actual URL
    # e.g., https://github.com/Kianjn/SmartPolicyAssistant.git
    git clone [URL of your GitHub repository]
    cd SmartPolicyAssistant
    ```

**3. Create Conda Environment:**
    *   **(Recommended: Using environment.yml)** If an `environment.yml` file is present in the repository:
        ```bash
        conda env create -f environment.yml
        conda activate policy_env
        ```
    *   **(Manual Steps)** Alternatively, follow these steps:
        ```bash
        conda create --name policy_env python=3.10 -y
        conda activate policy_env
        # Install FAISS/PyTorch (adjust for CPU/GPU - this is for CPU)
        conda install pytorch cpuonly faiss-cpu -c pytorch -c conda-forge -y
        # Install other dependencies
        pip install --upgrade pip
        pip install langchain langchain-google-genai google-generativeai langchain-community sentence-transformers python-dotenv streamlit faiss-cpu # Ensure faiss-cpu matches conda install if manual
        ```
    *   *(You may want to generate an `environment.yml` file after successful manual installation using `conda env export > environment.yml`)*

**4. Set Up Google API Key:**
    *   Create a file named `.env` inside the `02.RAG` directory.
    *   Add your Google API key to the file:
        ```bash
        # In 02.RAG/.env
        GOOGLE_API_KEY="YOUR_ACTUAL_GOOGLE_API_KEY_HERE"
        ```

**5. Prepare Data:**
    *   Obtain the NECP documents for the desired EU countries (e.g., from the European Commission website).
    *   Process these PDFs into plain `.txt` files using your scripts from the `01.Summarization` stage (or manually).
    *   Place the resulting `.txt` files inside the `02.RAG/necp_texts/` directory. Name them appropriately (e.g., `germany.txt`, `france.txt`).

**6. Create FAISS Index:**
    *   Navigate to the RAG directory: `cd 02.RAG`
    *   Run the indexing script. This needs to be done only once initially or when the `.txt` files change. You may need to edit `policy_qa.py` to set `force_recreate_vs=True` for the first run.
        ```bash
        # Check policy_qa.py settings if needed
        python policy_qa.py
        ```
    *   Verify that the `02.RAG/FAISS/` directory is created and contains `index.faiss` and `index.pkl`.
    *   **(Optional)** Set `force_recreate_vs=False` in `policy_qa.py` for subsequent runs if you only want to use the existing index.

## Usage

**1. Activate Environment:**
    ```bash
    conda activate policy_env
    ```

**2. Run the Streamlit Chat UI:**
    *   Navigate to the UI directory: `cd ../03.UI` (if you were in 02.RAG) or `cd 03.UI` (if you are in the project root `SmartPolicyAssistant/`)
    *   Start the Streamlit app:
        ```bash
        streamlit run ui_chat.py
        ```
    *   Open the URL provided in your terminal (usually `http://localhost:8501`) in your web browser.
    *   Ask questions about the NECP documents in the chat interface.

**Example Questions:**

*   "What is Germany's target for renewable energy share in 2030?"
*   "Summarize Italy's main policies for the transport sector."
*   "Compare the greenhouse gas emission reduction targets of France and Spain for 2030."
*   "What measures does Austria mention regarding energy efficiency in buildings?"

**3. (Optional) Run Command-Line Chat:**
    *   Navigate to the RAG directory: `cd ../02.RAG`
    *   Run the command-line chat script:
        ```bash
        python chat_with_policy.py
        ```

## Future Work / Potential Improvements

*   **Metadata Filtering:** Enhance retrieval by filtering based on country metadata attached to chunks.
*   **Advanced RAG Techniques:** Explore Multi-Query Retriever, MMR, Parent Document Retriever, or HyDE for better context retrieval, especially for comparative questions.
*   **Enhanced Comparison Logic:** Implement more sophisticated prompting or chain structures specifically designed for comparison tasks.
*   **Source Highlighting:** Modify the UI to optionally show which parts of which source documents were used to generate the answer.
*   **Evaluation Framework:** Implement metrics to evaluate the quality and accuracy of the RAG system's responses.
*   **Include Summarization:** Integrate the summarization capability from Step 1 directly into the workflow/UI.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

*   Data Sources: European Commission / Member State NECP documents.
*   Key Libraries: LangChain, Streamlit, FAISS, Google Generative AI.
