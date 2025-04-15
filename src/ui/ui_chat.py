# --- ui_chat.py (Simplified: No FAISS, CSV RAG UI with LLM Query Parsing) ---

# Import the Streamlit library for creating web apps
import streamlit as st
# Import standard Python libraries
import os
import sys # System-specific parameters and functions (not strictly needed here now, but often useful)
# Import library for loading environment variables
from dotenv import load_dotenv
# Import data manipulation library
import pandas as pd

# Import LangChain components for LLM interaction
try:
    from langchain_google_genai import ChatGoogleGenerativeAI # Class for Google Gemini models
    from langchain.prompts import PromptTemplate            # For creating structured prompts
    from langchain.chains import LLMChain                   # For running LLM with a prompt
except ImportError:
    # Display an error in the Streamlit app if LangChain is not installed
    st.error("Required LangChain libraries not found. Make sure you are running this in the correct Conda environment ('policy_env') and installed dependencies.")
    # Stop the Streamlit app execution if dependencies are missing
    st.stop()

# --- Path Setup & Env Loading ---
# Get the absolute path of the directory containing this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Determine the project root directory (assuming script is in root/src/ui or similar)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
# Define the path to the .env file and load environment variables
# Checks common locations: src/core/.env or project_root/.env
dotenv_path = os.path.join(PROJECT_ROOT, "src", "core", ".env")
if not os.path.exists(dotenv_path): dotenv_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(dotenv_path=dotenv_path) # Load variables from the file
print(f"Attempting to load .env from: {dotenv_path}") # Log path for debugging

# Retrieve the Google API Key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Constants ---
# Define the path to the aggregated CSV data file
AGGREGATED_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "aggregated_necp_data.csv")
# Specify the LLM model name to use
# GENAI CAPABILITY: Model Selection
LLM_MODEL = "gemini-1.5-flash-latest"

# --- Error Checks ---
# Check if the API key was loaded; stop the app if not
if not GOOGLE_API_KEY:
    st.error("Error: GOOGLE_API_KEY not found.")
    st.info(f"Please ensure a .env file with the key exists (checked {dotenv_path}).")
    st.stop()
# Check if the aggregated CSV data file exists; stop the app if not
if not os.path.exists(AGGREGATED_CSV_PATH):
    st.error(f"Error: Aggregated data file not found: {AGGREGATED_CSV_PATH}")
    st.info("Please ensure 'aggregate_summaries.py' ran successfully.")
    st.stop()

# --- Caching Functions ---
# Use Streamlit's caching for data loading to improve performance.
# @st.cache_data is suitable for data like DataFrames that don't change often.
@st.cache_data(show_spinner="Loading Policy Database...")
def load_aggregated_data(_csv_path: str) -> pd.DataFrame | None:
    """Loads the aggregated data CSV into a DataFrame, cached by Streamlit."""
    try:
        # Read the CSV file using Pandas
        df = pd.read_csv(_csv_path)
        # Show a success message in the Streamlit UI sidebar (can be moved)
        st.success(f"Policy Database loaded ({df.shape[0]} countries/rows).")
        return df # Return the loaded DataFrame
    except Exception as e:
        # Show an error message in the UI if loading fails
        st.error(f"Failed to load aggregated data from {_csv_path}: {e}")
        return None # Return None on failure

# Use Streamlit's caching for resources like LLM connections.
# @st.cache_resource ensures the LLM components are initialized only once per session.
@st.cache_resource(show_spinner="Initializing AI Assistant...")
def initialize_llm_components() -> tuple[ChatGoogleGenerativeAI | None, LLMChain | None]:
    """Initializes the LLM instance and the Synthesizer Chain, cached by Streamlit."""
    # Double-check API key just in case
    if not GOOGLE_API_KEY: return None, None
    try:
        # Initialize the LLM instance (used for both parsing and synthesis)
        # GENAI CAPABILITY: Accessing Foundational Model (Gemini)
        # GENAI CAPABILITY: Controlled Generation (temperature setting)
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.1) # Low temp good for parsing

        # Define the prompt template for the answer synthesis step
        # GENAI CAPABILITY: Prompt Engineering (Defining role, task, constraints for synthesis)
        synth_template = """
You are a helpful assistant analyzing National Energy and Climate Plan (NECP) data.
Answer the following question based *only* on the structured data provided below.
Present the answer clearly and concisely in natural language. If the data allows for comparison, make the comparison. If the data seems insufficient to fully answer, say so.

Question: {question}

Data:
{data_context}

Answer:"""
        # Create the PromptTemplate object
        synth_prompt = PromptTemplate(template=synth_template, input_variables=["question", "data_context"])
        # Create the LLMChain for synthesis
        synthesizer_chain = LLMChain(prompt=synth_prompt, llm=llm)

        # Show an info message in the UI
        st.info("AI Assistant ready.")
        # Return the initialized LLM instance and the synthesizer chain
        return llm, synthesizer_chain
    except Exception as e:
        # Show error in UI if initialization fails
        st.error(f"Failed to initialize AI Assistant (LLM Chain): {e}")
        return None, None # Return None values on failure

# --- LLM Function to Parse Question for Columns ---
# This function is defined here but could be moved to a shared utility file.
# It is NOT cached with st.cache_resource because it depends on the varying 'question' input.
def parse_question_for_columns(question: str, available_columns: list[str], llm: ChatGoogleGenerativeAI | None) -> list[str] | None:
    """Uses LLM to identify relevant columns for a question."""
    # Check if LLM instance is valid
    if not llm: return None
    # Format the column list for the prompt
    column_list_str = "\n - ".join(available_columns)
    # Define the prompt for the column parsing task
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
    # Log the attempt (useful for server logs when deployed)
    print(f"\nDEBUG: Sending to LLM for Column Parsing. Question: '{question}'")
    try:
        # Invoke the LLM with the parsing prompt
        # GENAI CAPABILITY: Calling Foundational Model (Gemini API call for parsing)
        response = llm.invoke(prompt_text)
        # Extract and clean the response content
        raw_result = response.content.strip()
        print(f"DEBUG: LLM Raw Column Response: '{raw_result}'") # Server log
        # Handle "NONE" response
        if raw_result.upper() == "NONE" or not raw_result: return []
        # Parse the comma-separated list
        parsed_columns = [col.strip() for col in raw_result.split(',')]
        # Validate parsed columns against the available list
        valid_columns = [col for col in parsed_columns if col in available_columns]
        print(f"DEBUG: Validated Relevant Columns: {valid_columns}") # Server log
        return valid_columns # Return the validated list
    except Exception as e:
        # Log errors during parsing
        print(f"Error during LLM column parsing: {e}")
        # Optionally show error in UI? st.warning(...)
        return None # Indicate error

# --- CSV Query Function (Uses LLM Parser) ---
# This function orchestrates the retrieval step of the CSV RAG.
def find_relevant_data_in_csv(df: pd.DataFrame, question: str, llm_parser_instance: ChatGoogleGenerativeAI | None) -> pd.DataFrame | str | None:
    """
    Identifies relevant countries, calls LLM to identify columns, and filters the DataFrame.
    Returns a DataFrame subset, a help/info message string, or None on error.
    """
    # Basic checks
    if df is None: return None
    if llm_parser_instance is None: return "Error: LLM Parser is not available."
    question_lower = question.lower() # Use lowercase for matching
    try:
        # 1. Identify mentioned countries (basic keyword check)
        mentioned_countries = [country for country in df['country'].unique() if country.lower() in question_lower]
        if not mentioned_countries: return "Help: Please specify at least one country." # Return help message

        # 2. Call the LLM parser to get relevant columns
        available_columns = df.columns.tolist()
        # GENAI CAPABILITY: Document Understanding (LLM understands the question to find columns)
        parsed_columns = parse_question_for_columns(question, available_columns, llm_parser_instance)

        # 3. Handle parsing results and determine final columns
        if parsed_columns is None: return "Error: Failed to determine relevant data columns." # Error during parsing
        if not parsed_columns: # LLM indicated no specific relevant columns
             print("Info: No specific columns identified by LLM, using default targets.") # Log this case
             # Fallback to default columns if only countries were mentioned
             relevant_columns = ['country', 'ghg_target_2030', 'renewable_target_2030', 'efficiency_target_2030']
        else:
             # Use LLM's suggestions, ensuring 'country' is always present
             relevant_columns = ['country'] + [col for col in parsed_columns if col != 'country']

        # 4. Filter the DataFrame
        # Validate column names against the actual DataFrame columns
        final_columns = [col for col in relevant_columns if col in df.columns]
        # Check if any valid data columns remain
        if len(final_columns) <= 1 : return f"Info: Could not find relevant data columns matching query for {', '.join(mentioned_countries)}." # Info message

        # Perform the filtering
        filtered_df = df[df['country'].isin(mentioned_countries)][final_columns]

        # Check if the result is empty
        if filtered_df.empty: return f"Info: No specific data found for criteria for {', '.join(mentioned_countries)}." # Info message

        # Success: return the filtered DataFrame
        return filtered_df
    except Exception as e:
        # Show error in UI if query fails
        st.error(f"Error during DataFrame query: {e}", icon="⚠️")
        return None # Indicate error


# --- Main Streamlit App UI and Logic ---
# Set page configuration (title, layout)
st.set_page_config(page_title="Smart Policy Assistant", layout="wide")
# Display the main title
st.title("🇪🇺 Smart Policy Assistant Chat (LLM Query Parsing)")
# Display an informational message below the title
st.info("Ask questions about EU countries' NECP data (e.g., 'Compare Germany and France ghg_target_2030').")

# --- Load Data & Initialize LLM Components using Cached Functions ---
# Attempt to load the aggregated CSV data
df_aggregated = load_aggregated_data(AGGREGATED_CSV_PATH)
# Attempt to initialize the LLM instance (parser) and the synthesizer chain
llm_parser_instance, llm_synthesizer_chain = initialize_llm_components()

# --- Critical Check ---
# Stop the app if essential components (data or LLM chains) failed to load/initialize
if df_aggregated is None or llm_parser_instance is None or llm_synthesizer_chain is None:
    st.error("Core components failed to load. Cannot start chat. Please check logs or API key.")
    st.stop() # Halt execution

# --- Chat History Management ---
# Initialize the chat history in Streamlit's session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages from the history on app rerun
for message in st.session_state.messages:
    # Use Streamlit's chat_message context manager to display messages with roles (user/assistant)
    with st.chat_message(message["role"]):
        # Display the message content as Markdown (allows table formatting etc.)
        st.markdown(message["content"], unsafe_allow_html=True) # Allow HTML for markdown tables

# --- Handle User Input ---
# Use Streamlit's chat_input widget to get user input at the bottom of the page
if prompt := st.chat_input("Ask about policy data..."):
    # Append the user's message to the session state chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display the user's message immediately in the chat interface
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Generate and Display Assistant Response ---
    # Use the assistant's chat message context manager
    with st.chat_message("assistant"):
        # Create a placeholder to show "Thinking..." messages while processing
        message_placeholder = st.empty()
        # Initial "Thinking" message
        message_placeholder.markdown("Understanding question...")

        # 1. Retrieve structured data context using LLM Parser
        # GENAI CAPABILITY: Retrieval augmented generation (RAG) - adapted for structured data.
        # Step 1: Retrieve - Calling `find_relevant_data_in_csv`.
        retrieved_context = find_relevant_data_in_csv(df_aggregated, prompt, llm_parser_instance)

        # 2. Handle retrieval results and generate final answer string
        final_answer = "" # Initialize empty answer string
        if retrieved_context is None: # Error during retrieval
            final_answer = "An error occurred while trying to retrieve relevant data."
        elif isinstance(retrieved_context, str): # Help/Info message returned
            final_answer = retrieved_context.replace("Help: ", "").replace("Info: ", "")
        elif isinstance(retrieved_context, pd.DataFrame): # Data successfully retrieved
            # Step 2: Augment - Format the DataFrame for the LLM synthesizer prompt
            formatted_df = retrieved_context.fillna("Not specified") # Handle NaNs
            data_context_str = formatted_df.to_markdown(index=False) # Convert to Markdown

            # Basic length check (prevent sending extremely large context)
            # TODO: Implement proper token counting for more accuracy
            if len(data_context_str) > 8000: # Arbitrary character limit
                data_context_str = "Retrieved data is too large. Please ask a more specific question."
                final_answer = data_context_str # Return this message directly
            else:
                # Update placeholder message before calling synthesizer LLM
                message_placeholder.markdown("Synthesizing answer...")
                # Step 3: Generate - Call the synthesizer LLM chain
                # GENAI CAPABILITY: Calling Foundational Model (Gemini API call for synthesis)
                try:
                    response = llm_synthesizer_chain.invoke({"question": prompt, "data_context": data_context_str})
                    final_answer = response.get('text', "Sorry, I couldn't generate an answer based on the data.")
                except Exception as e:
                    # Show error in UI if synthesis fails
                    st.error(f"Error during LLM synthesis: {e}", icon="🤖")
                    final_answer = "An error occurred during answer generation."
        else: # Should not happen
            final_answer = "Unexpected error after data retrieval."

        # Display the final generated answer in the placeholder
        # Use unsafe_allow_html=True to render markdown tables correctly
        message_placeholder.markdown(final_answer, unsafe_allow_html=True)
        # Add the assistant's final answer to the chat history
        st.session_state.messages.append({"role": "assistant", "content": final_answer})

# --- Sidebar ---
# Add a separator in the sidebar
st.sidebar.markdown("---")
# Add a header for the About section
st.sidebar.header("About")
# Provide a brief description of the assistant
st.sidebar.info("This assistant uses AI to understand your question and retrieve relevant data from the aggregated NECP database.")
# If the DataFrame is loaded, add an expander to show available columns
if df_aggregated is not None:
    with st.sidebar.expander("Show Available Data Fields"):
         # Display the column names in a simple DataFrame for clarity
         st.dataframe(pd.DataFrame(df_aggregated.columns, columns=["Field Name"]), use_container_width=True)

# --- GenAI Capability Analysis Summary ---
# - Retrieval augmented generation (RAG): Adapted form - Retrieves from structured CSV using LLM parsing, then generates answer using LLM synthesis.
# - Document Understanding: LLM in `parse_question_for_columns` understands the user's natural language query.
# - Structured output: `parse_question_for_columns` requests a comma-separated list or "NONE".
# - Prompt Engineering: Used for both column parsing and answer synthesis prompts.
# - Foundational Model Access: Uses `ChatGoogleGenerativeAI` to call the Gemini model for parsing and synthesis.
# - Controlled Generation: Temperature parameter set in `ChatGoogleGenerativeAI`.
# - Context Caching: Streamlit's `@st.cache_data` and `@st.cache_resource` are used to cache data loading and LLM initialization, improving performance on reruns.
# - (Implicit) Long context window: The synthesizer might receive large context.
# - (Not Used: FAISS/Vector Search (explicitly removed), Few-shot prompting, Image/Video/Audio understanding, Function Calling, Agents, Gen AI evaluation, Grounding, Embeddings, MLOps)