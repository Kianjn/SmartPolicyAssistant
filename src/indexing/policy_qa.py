# --- policy_qa.py (Simplified: No FAISS, CSV RAG CLI Test with LLM Query Parsing) ---

# Import necessary standard Python libraries
import os                         # For interacting with the operating system (paths)
import re                         # For regular expressions (used for basic country name matching)
import sys                        # For exiting the script cleanly on errors (sys.exit)

# Import library for loading environment variables
from dotenv import load_dotenv    # Used to load API keys from a .env file

# Import data manipulation library
import pandas as pd               # Used for loading and manipulating the aggregated CSV data

# Import LangChain components for interacting with Large Language Models (LLMs)
from langchain_google_genai import ChatGoogleGenerativeAI # Specific class for Google Gemini models
from langchain.prompts import PromptTemplate            # For creating reusable prompt structures
from langchain.chains import LLMChain                   # Simple chain for running LLM with a prompt
# LangChain Core components (Optional, could be used for more complex chains)
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# --- Configuration ---
# Get the absolute path of the directory where the current script resides
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Calculate the project's root directory (assuming script is two levels down from root, e.g., root/src/core)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

# Define the path to the .env file and load environment variables
# Checks common locations: src/core/.env or project_root/.env
dotenv_path = os.path.join(PROJECT_ROOT, "src", "core", ".env")
if not os.path.exists(dotenv_path): dotenv_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(dotenv_path=dotenv_path) # Load variables from the file
print(f"Attempting to load .env from: {dotenv_path}") # Log the path used

# Retrieve the Google API Key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Check if the API key was loaded; exit if not, as it's required for LLM calls
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found.")
    sys.exit(1) # Exit script if API key is missing

# --- Constants ---
# Define the path to the aggregated CSV data file (output from aggregate_summaries.py)
AGGREGATED_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "aggregated_necp_data.csv")
# Specify the LLM model name to use (from Google Generative AI)
# GENAI CAPABILITY: Model Selection
LLM_MODEL = "gemini-1.5-flash-latest" # Using flash for speed, but pro might be better for complex tasks

# --- Global Variables ---
# Initialize global variables to hold the loaded data and LLM components
df_aggregated = None          # Will hold the Pandas DataFrame loaded from CSV
llm_parser = None             # Will hold the LLM instance used for parsing questions
llm_synthesizer_chain = None  # Will hold the LangChain chain used for synthesizing final answers

# --- CSV Data Loading Function ---
def load_aggregated_data():
    """Loads the aggregated data CSV into the global DataFrame."""
    global df_aggregated
    # Return immediately if data is already loaded
    if df_aggregated is not None: return df_aggregated
    # Log the loading process
    print(f"Loading aggregated data from: {AGGREGATED_CSV_PATH}")
    # Check if the CSV file exists
    if not os.path.exists(AGGREGATED_CSV_PATH):
         print(f"ERROR: Aggregated data file not found: {AGGREGATED_CSV_PATH}")
         return None # Return None if file not found
    # Use try-except to handle potential errors during file reading/parsing
    try:
        # Read the CSV file into a Pandas DataFrame
        df_aggregated = pd.read_csv(AGGREGATED_CSV_PATH)
        # Log success and DataFrame dimensions
        print(f"Aggregated data loaded: {df_aggregated.shape[0]} rows, {df_aggregated.shape[1]} columns.")
        # Print the list of available column names for user reference
        print("Available data columns:", df_aggregated.columns.tolist())
        return df_aggregated # Return the loaded DataFrame
    except Exception as e:
        # Log any errors encountered during loading
        print(f"ERROR: Failed to load aggregated data: {e}")
        return None # Return None on error


# --- LLM Initialization Function ---
def initialize_llm_components():
    """Initializes the LLM instance and the synthesis chain."""
    global llm_parser, llm_synthesizer_chain
    # Return immediately if components are already initialized
    if llm_parser and llm_synthesizer_chain: return True
    print("Initializing LLM Components...")
    try:
        # Create a single LLM instance to be used for both parsing and synthesis
        # GENAI CAPABILITY: Accessing Foundational Model (Gemini)
        # GENAI CAPABILITY: Controlled Generation (temperature=0.1 for more deterministic parsing)
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.1)
        # Assign the same LLM instance to the parser variable
        llm_parser = llm

        # Define the prompt template for the answer synthesis step
        # GENAI CAPABILITY: Prompt Engineering (Defining role, task, constraints, context injection)
        synth_template = """
You are a helpful assistant analyzing National Energy and Climate Plan (NECP) data.
Answer the following question based *only* on the structured data provided below.
Present the answer clearly and concisely in natural language. If the data allows for comparison, make the comparison.

Question: {question}

Data:
{data_context}

Answer:"""
        # Create a PromptTemplate object from the template string
        synth_prompt = PromptTemplate(template=synth_template, input_variables=["question", "data_context"])
        # Create an LLMChain for the synthesis task, combining the prompt and the LLM instance
        llm_synthesizer_chain = LLMChain(prompt=synth_prompt, llm=llm)
        print("LLM Parser and Synthesizer Chain ready.")
        return True # Indicate successful initialization
    except Exception as e:
        # Log any errors during LLM initialization
        print(f"Failed to initialize LLM Components: {e}")
        # Reset global variables on failure
        llm_parser = None
        llm_synthesizer_chain = None
        return False # Indicate failure


# --- LLM Function to Parse Question for Columns ---
def parse_question_for_columns(question: str, available_columns: list[str], llm: ChatGoogleGenerativeAI) -> list[str] | None:
    """Uses LLM to identify relevant columns from a list based on the user's question."""
    # Check if the LLM instance is available
    if not llm:
        print("Error: LLM Parser not initialized.")
        return None

    # Format the list of available columns for inclusion in the prompt
    column_list_str = "\n - ".join(available_columns)
    # Define the prompt specifically for the column parsing task
    # GENAI CAPABILITY: Prompt Engineering (Task-specific instructions for column extraction)
    # GENAI CAPABILITY: Structured output (Requesting comma-separated list or "NONE")
    prompt_text = f"""
Given the user question and the following list of available data columns, identify which columns are most relevant to answering the question.
Return ONLY a comma-separated list of the relevant column names from the provided list. Do not include the 'country' column unless specifically asked for.
If no columns from the list seem relevant, return the word NONE.

Available Columns:
 - {column_list_str}

User Question: "{question}"

Relevant Column Names (comma-separated):"""

    # Log the parsing attempt for debugging
    print(f"\nDEBUG: Sending to LLM for Column Parsing. Question: '{question}'")
    try:
        # Make the API call to the LLM instance with the parsing prompt
        # GENAI CAPABILITY: Calling Foundational Model (Gemini API call for parsing)
        response = llm.invoke(prompt_text)
        # Extract the text content from the LLM's response and remove leading/trailing whitespace
        raw_result = response.content.strip()
        # Log the raw response from the LLM for debugging
        print(f"DEBUG: LLM Raw Column Response: '{raw_result}'")

        # Check if the LLM indicated no relevant columns were found
        if raw_result.upper() == "NONE" or not raw_result:
            return [] # Return an empty list

        # Parse the comma-separated string into a list of potential column names
        parsed_columns = [col.strip() for col in raw_result.split(',')]

        # **Crucial Validation Step:** Filter the parsed columns to ensure they actually exist in the original list.
        # This prevents the LLM from hallucinating column names.
        valid_columns = [col for col in parsed_columns if col in available_columns]

        # Log the validated columns for debugging
        print(f"DEBUG: Validated Relevant Columns: {valid_columns}")
        return valid_columns # Return the list of valid, relevant column names

    except Exception as e:
        # Log any errors that occur during the LLM call or parsing
        print(f"Error during LLM column parsing: {e}")
        return None # Return None to indicate an error occurred


# --- CSV Query Function (Uses LLM Parser) ---
def find_relevant_data_in_csv(df: pd.DataFrame, question: str, llm_parser_instance: ChatGoogleGenerativeAI) -> pd.DataFrame | str | None:
    """
    Identifies relevant countries and columns (using LLM parser), then filters the DataFrame.
    Returns a DataFrame subset, a help/info message string, or None on error.
    """
    # Check if DataFrame and LLM parser are available
    if df is None: return None
    if llm_parser_instance is None: return "Error: LLM Parser is not available."

    question_lower = question.lower() # Convert question to lowercase for matching
    try:
        # 1. Identify mentioned countries using simple string matching (case-insensitive)
        mentioned_countries = [country for country in df['country'].unique() if country.lower() in question_lower]
        # Check if at least one country was mentioned
        if not mentioned_countries:
            return "Help: Please specify at least one country in your question." # Return help message

        # 2. Call the LLM parser function to get relevant columns
        available_columns = df.columns.tolist()
        parsed_columns = parse_question_for_columns(question, available_columns, llm_parser_instance)

        # Handle results from the column parser
        if parsed_columns is None: # An error occurred during parsing
            return "Error: Failed to determine relevant data columns using AI."
        if not parsed_columns: # LLM returned NONE or only invalid columns
             # If no specific columns were identified, fall back to default key targets
             print("Info: No specific columns identified by LLM, using default targets.")
             relevant_columns = ['country', 'ghg_target_2030', 'renewable_target_2030', 'efficiency_target_2030']
        else:
             # Use the columns identified by the LLM, ensuring 'country' is always included
             relevant_columns = ['country'] + [col for col in parsed_columns if col != 'country']


        # 3. Filter the DataFrame based on countries and identified columns
        # Ensure only columns that actually exist in the DataFrame are used (safety check)
        final_columns = [col for col in relevant_columns if col in df.columns]
        # Check if any relevant data columns remain after validation
        if len(final_columns) <= 1 : # Only 'country' column is left or none were valid
             return f"Info: Could not find relevant data columns matching your query for {', '.join(mentioned_countries)}." # Return info message

        # Perform the filtering using Pandas .isin() for countries and column selection
        filtered_df = df[df['country'].isin(mentioned_countries)][final_columns]

        # Check if the filtered DataFrame is empty (no matching data found)
        if filtered_df.empty:
            return f"Info: No specific data found for the requested criteria for {', '.join(mentioned_countries)}." # Return info message

        # If data is found, return the filtered DataFrame subset
        return filtered_df

    except Exception as e:
        # Log any unexpected errors during the query process
        print(f"Error during DataFrame query: {e}")
        return None # Return None to indicate an error


# --- Main Q&A Function ---
def ask_policy(question: str):
    """
    Orchestrates the process: parses question, retrieves data, synthesizes answer.
    """
    global df_aggregated, llm_parser, llm_synthesizer_chain
    # Check if essential components are loaded
    if df_aggregated is None: return {"answer": "Error: Aggregated data not loaded.", "source": "Error"}
    if llm_parser is None: return {"answer": "Error: LLM Parser not initialized.", "source": "Error"}
    if llm_synthesizer_chain is None: return {"answer": "Error: LLM Synthesizer not initialized.", "source": "Error"}

    # Log the question being processed
    print(f"\nProcessing question: {question}")
    # Call the function to retrieve data, passing the LLM parser instance
    # GENAI CAPABILITY: Retrieval augmented generation (RAG) - adapted for structured data.
    # Step 1: Retrieve - Querying CSV using LLM-parsed columns via `find_relevant_data_in_csv`.
    retrieved_context = find_relevant_data_in_csv(df_aggregated, question, llm_parser)

    # --- Handle Retrieval Results and Synthesize Answer ---
    answer = ""
    source = "" # To indicate the source/status of the answer

    if retrieved_context is None: # Error during retrieval
        answer = "An error occurred while trying to retrieve data from the CSV."
        source = "Error"
    elif isinstance(retrieved_context, str): # Help/Info message from retrieval
        answer = retrieved_context.replace("Help: ", "").replace("Info: ", "") # Clean prefix
        source = "Info Message"
    elif isinstance(retrieved_context, pd.DataFrame): # Successful retrieval (got DataFrame)
        # Step 2: Augment - Format the retrieved DataFrame for the LLM prompt.
        # Handle missing values (NaN) before converting to markdown
        formatted_df = retrieved_context.fillna("Not specified")
        # Convert the DataFrame to a markdown string (often good for LLM context)
        data_context_str = formatted_df.to_markdown(index=False)
        # Log the formatted context being sent to the synthesizer LLM (for debugging)
        print(f"DEBUG: Context for LLM Synthesizer:\n{data_context_str}")

        # Basic check for context length to avoid overwhelming the LLM or hitting limits
        # NOTE: A proper solution would count tokens, not characters.
        if len(data_context_str) > 8000: # Use a reasonable character limit (adjust as needed)
             data_context_str = "Retrieved data is too large to synthesize effectively. Please ask a more specific question."
             answer = data_context_str
             source = "Info Message (Context Too Large)"
        else:
            # Step 3: Generate - Call the synthesizer LLM chain.
            # GENAI CAPABILITY: Calling Foundational Model (Gemini API call for synthesis)
            try:
                # Pass the original question and the formatted data context to the chain
                response = llm_synthesizer_chain.invoke({"question": question, "data_context": data_context_str})
                # Extract the synthesized text answer from the response
                answer = response.get('text', "Sorry, I couldn't generate an answer based on the retrieved data.")
                source = "CSV RAG" # Indicate successful answer generation from CSV data
            except Exception as e:
                # Handle errors during the LLM synthesis call
                print(f"Error during LLM synthesis: {e}")
                answer = "An error occurred during answer generation."
                source = "Error"
    else: # Should not happen if find_relevant_data_in_csv returns correctly
        answer = "Unexpected error during data retrieval process."
        source = "Error"

    # --- Print the Final Answer and Source ---
    print("\n--- Answer ---")
    print(answer)
    print(f"(Source: {source})") # Indicate how the answer was derived
    print("--------------")
    # Return a dictionary containing the answer and its source/status
    return {"answer": answer, "source": source}


# --- Main Execution Block (for CLI Testing) ---
# This code runs only when the script is executed directly
if __name__ == "__main__":
    print("--- Policy QA Initializing (LLM Query Parsing CLI Test) ---")
    # 1. Load Aggregated Data (Mandatory)
    print("\nInitializing Aggregated Data (CSV)...")
    load_aggregated_data()
    # Exit if data loading failed
    if df_aggregated is None: sys.exit("Exiting: Failed to load aggregated data.")

    # 2. Initialize LLM Components (Parser and Synthesizer - Mandatory)
    print("\nInitializing LLM Components...")
    # Exit if LLM initialization failed (e.g., missing API key)
    if not initialize_llm_components():
        sys.exit("Exiting: Failed to initialize LLM components.")

    # --- Start Interactive Command-Line Loop ---
    print("\n--- Policy Data Q&A (LLM Query Parsing) ---")
    print("Ask questions about the aggregated NECP data. Type 'quit' or 'exit' to stop.")
    # Loop indefinitely until user quits
    while True:
        try:
            # Prompt the user for input
            user_question = input("Ask: ")
            # Check for exit commands
            if user_question.lower().strip() in ['quit', 'exit']: break
            # Ignore empty input
            if not user_question.strip(): continue
            # Call the main Q&A function to process the question and print the result
            ask_policy(user_question)
        # Handle graceful exit on Ctrl+D (EOFError) or Ctrl+C (KeyboardInterrupt)
        except (EOFError, KeyboardInterrupt): print("\nExiting..."); break
    # Print closing message when the loop ends
    print("--- Session Ended ---")

# --- GenAI Capability Analysis Summary ---
# - Retrieval augmented generation (RAG): Adapted form - Retrieves from structured CSV using LLM parsing, then generates answer using LLM.
# - Document Understanding: LLM in `parse_question_for_columns` understands the user's natural language query.
# - Structured output: `parse_question_for_columns` requests a comma-separated list or "NONE".
# - Prompt Engineering: Used for both column parsing and answer synthesis prompts.
# - Foundational Model Access: Uses `ChatGoogleGenerativeAI` to call the Gemini model for two distinct tasks (parsing, synthesis).
# - Controlled Generation: Temperature parameter set in `ChatGoogleGenerativeAI`.
# - (Implicit) Long context window: The synthesizer might receive large context if many columns/countries are retrieved.
# - (Not Used: FAISS/Vector Search (explicitly removed), Few-shot prompting, Image/Video/Audio understanding, Function Calling, Agents, Context caching, Gen AI evaluation, Grounding, Embeddings, MLOps)