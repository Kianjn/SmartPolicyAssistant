# Import necessary standard Python libraries
import os                     # For interacting with the operating system (paths, directories)
import json                   # For working with JSON data (loading/saving structured data)
import time                   # For pausing execution (e.g., API delays, retries)
import re                     # For regular expressions (used for filename sanitization, metadata extraction)
import random                 # For adding jitter (randomness) to retry delays
from collections import defaultdict # For easily grouping files by country

# Import progress bar library
from tqdm import tqdm         # For displaying progress bars during loops

# Import Google Generative AI specific libraries
import google.generativeai as genai # The main SDK for interacting with Gemini models

# Import specific exception types for handling API errors (like rate limits)
from google.api_core import exceptions as google_exceptions
# Import types for configuring safety settings for the Gemini model
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Import library for loading environment variables from a .env file
from dotenv import load_dotenv

# --- Configuration ---
# Determine the project's root directory based on the current script's location
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Define the path to the .env file (used for storing the API key securely)
# Check common locations: src/core/.env or project_root/.env
dotenv_path = os.path.join(PROJECT_ROOT, "src", "core", ".env")
if not os.path.exists(dotenv_path): dotenv_path = os.path.join(PROJECT_ROOT, ".env")
# Load the environment variables from the .env file
load_dotenv(dotenv_path=dotenv_path)
# Print the path being checked for the .env file (for debugging)
print(f"Attempting to load .env from: {dotenv_path}")

# Define the path to the folder containing the input text files (output from pdf_to_txt.py)
TXT_FOLDER = os.path.join(PROJECT_ROOT, "data", "processed")
# Define the path to the folder where the output JSON summary files will be saved
JSON_FOLDER = os.path.join(PROJECT_ROOT, "data", "processed", "summaries")

# --- Google AI Configuration ---
# Retrieve the Google API Key from environment variables (loaded from .env)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Specify the Gemini model to be used for generation (e.g., flash for speed/cost, pro for capability)
# GENAI CAPABILITY: Model Selection (Choosing appropriate GenAI model)
MODEL_NAME = "gemini-1.5-flash-latest"

# Configuration settings for the generation process
# GENAI CAPABILITY: Controlled Generation (via parameters)
GENERATION_CONFIG = {
    "max_output_tokens": 4096, # Set maximum number of tokens in the generated output (increased for large JSON)
    "temperature": 0.2,      # Lower temperature -> less random, more deterministic/factual output
    "top_p": 0.95,           # Nucleus sampling parameter (alternative to temperature)
}
# Configure safety settings to block potentially harmful content above specified thresholds
# GENAI CAPABILITY: Content Safety/Filtering
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    }

# --- Retry Configuration ---
# MLOps: Configuration for handling API errors (specifically rate limits)
MAX_RETRIES = 4              # Maximum number of times to retry an API call if it fails with a rate limit error
INITIAL_RETRY_DELAY = 15     # Base delay (in seconds) before the first retry attempt
MAX_RETRY_DELAY = 120        # Maximum wait time (in seconds) between retries, prevents excessively long waits
# Base delay (in seconds) to wait AFTER a successful API call for one country before starting the next.
# This helps prevent overwhelming the API with frequent large requests. Adjust based on observed limits.
BASE_API_DELAY_SUCCESS = 15.0

# --- Initialization ---
# Create the output JSON folder if it doesn't exist
os.makedirs(JSON_FOLDER, exist_ok=True)
# Check if the API key was loaded successfully; exit if not
if not GOOGLE_API_KEY: print("ERROR: GOOGLE_API_KEY is not set."); exit(1)
# Configure the Google AI SDK with the API key
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Google AI SDK configured successfully.")
except Exception as e:
    print(f"Error configuring Google AI SDK: {e}"); exit(1)
# Initialize the generative model instance using the specified model name
try:
    # GENAI CAPABILITY: Accessing Foundational Model (Gemini)
    model = genai.GenerativeModel(MODEL_NAME)
    print(f"Using Gemini model: {MODEL_NAME}")
except Exception as e:
    print(f"Error loading model '{MODEL_NAME}': {e}"); exit(1)


# --- Helper Functions ---
# Constant defining the end marker of the metadata block in the text files
METADATA_END_MARKER = "=================="

# Function to read a text file and extract the country name from its metadata block
def get_country_from_metadata(filepath):
    """Reads the metadata block and extracts the Country name."""
    try:
        # Open the file with UTF-8 encoding
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read only the first 1000 characters (assuming metadata is at the start)
            content = f.read(1000)
            # Use regex to find the line starting with "Country:" (case-insensitive) and capture the value
            match = re.search(r"^Country:\s*(.*?)\s*$", content, re.MULTILINE | re.IGNORECASE)
            if match:
                # If found, extract the captured group, strip whitespace
                country = match.group(1).strip()
                # Basic normalization: Remove " Federal Republic" (e.g., from Germany)
                country = country.replace(" Federal Republic", "")
                return country # Return the cleaned country name
            else:
                # Warn if the "Country:" line wasn't found
                print(f"Warning: Could not find 'Country:' in metadata for {os.path.basename(filepath)}")
                return None # Indicate failure to find country
    except Exception as e:
        # Handle errors during file reading or regex processing
        print(f"Error reading metadata from {os.path.basename(filepath)}: {e}")
        return None # Indicate error

# Function to read a text file and return only the content *after* the metadata block
def get_body_text(filepath):
     """Reads the file and returns content *after* the metadata block."""
     try:
        # Open the file with UTF-8 encoding
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read the entire file content
            content = f.read()
            # Find the position of the metadata end marker
            marker_pos = content.find(METADATA_END_MARKER)
            if marker_pos != -1:
                # If marker found, return the text slice starting *after* the marker and its trailing newline
                return content[marker_pos + len(METADATA_END_MARKER):].strip()
            else:
                # Warn if the marker wasn't found; return the whole content as a fallback
                print(f"Warning: Metadata end marker '{METADATA_END_MARKER}' not found in {os.path.basename(filepath)}. Returning full content.")
                return content
     except Exception as e:
        # Handle file reading errors
        print(f"Error reading body text from {os.path.basename(filepath)}: {e}")
        return "" # Return empty string on error

# Function to construct the detailed prompt for the Gemini model
def build_prompt(text_to_summarize, country):
    """Constructs the detailed prompt for the Gemini model."""
    # NOTE: This uses a large, detailed prompt with specific instructions.
    # GENAI CAPABILITY: Prompt Engineering (Detailed instructions, role-playing, format specification)
    # GENAI CAPABILITY: Structured output/JSON mode/controlled generation (Requesting specific JSON format)
    # GENAI CAPABILITY: Document Understanding (The LLM needs to read and comprehend the 'text_to_summarize')
    # (No few-shot examples included in this version, but the structure implies it could be added)
    prompt = f"""
You are an expert analyst specializing in National Energy and Climate Plans (NECPs).
Your task is to meticulously read the provided NECP text for {country} (which might consist of multiple concatenated parts) and extract the following specific information.
If a specific piece of information is not found or clearly stated, return the string "Not specified". Do not make up information.
Extract quantitative targets with their units (like %, MW, GW, year) and base years/comparison points if mentioned.

Return ONLY the extracted information in a single, valid JSON object, adhering strictly to the structure below.
Do not include any introductory text, explanations, apologies, markdown formatting (like ```json), or anything else before or after the JSON object itself.


JSON Format:
{{
  "country": "{country}",
  "ghg_target_2030": "<Overall GHG emission reduction target for 2030, including base year or comparison point, e.g., '55% reduction from 1990 levels' or 'Not specified'>",
  "renewable_target_2030": "<Overall renewable energy share target for 2030, including scope (e.g., 'gross final energy consumption', 'electricity'), e.g., '40% share in gross final energy consumption' or 'Not specified'>",
  "efficiency_target_2030": "<Overall energy efficiency target for 2030, including scope (primary/final energy) and comparison point, e.g., '35% reduction in primary energy consumption vs baseline' or 'Not specified'>",
  "carbon_pricing": "<Strictly 'Yes' or 'No' if explicit carbon pricing (ETS, tax) is mentioned as a policy measure, default to 'No' if unsure or not mentioned>",
  "measures": ["<List of 5-7 key distinct policy measures mentioned, e.g., 'Phase-out coal by 2038', 'Carbon tax on transport', 'Building renovation strategy' or [] if none found>"],

  "renewable_target_electricity_2030": "<Renewable energy target specifically for the electricity sector by 2030, e.g., '65% share in electricity generation' or 'Not specified'>",
  "renewable_target_heating_cooling_2030": "<Renewable energy target specifically for the heating and cooling sector by 2030, e.g., '30% share' or 'Not specified'>",
  "renewable_target_transport_2030": "<Renewable energy target specifically for the transport sector by 2030, e.g., '14% share' or 'Not specified'>",
  "planned_capacity_solar_pv_2030": "<Planned installed capacity for Solar PV by 2030, including units (MW or GW), e.g., '50 GW' or 'Not specified'>",
  "planned_capacity_onshore_wind_2030": "<Planned installed capacity for Onshore Wind by 2030, including units (MW or GW), e.g., '70 GW' or 'Not specified'>",
  "planned_capacity_offshore_wind_2030": "<Planned installed capacity for Offshore Wind by 2030, including units (MW or GW), e.g., '30 GW' or 'Not specified'>",
  "planned_capacity_hydro_2030": "<Planned installed capacity for Hydropower by 2030, including units (MW or GW), e.g., 'Maintain current levels' or 'Not specified'>",
  "planned_capacity_geothermal_2030": "<Planned installed capacity for Geothermal energy by 2030, including units (MW or GWh), e.g., '100 MW' or 'Not specified'>",
  "hydrogen_strategy_mentioned": "<'Yes', 'No', or brief details if a specific hydrogen strategy or significant focus is mentioned, e.g., 'Yes, focus on green hydrogen for industry' or 'No'>",
  "hydrogen_production_target": "<Specific production target for hydrogen (mention type like green/blue if specified) by a certain year, e.g., '5 GW electrolysis capacity by 2030' or 'Not specified'>",

  "primary_energy_consumption_target_2030": "<Target for Primary Energy Consumption by 2030 (absolute value or reduction percentage), e.g., 'Limit to 100 Mtoe' or 'Not specified'>",
  "final_energy_consumption_target_2030": "<Target for Final Energy Consumption by 2030 (absolute value or reduction percentage), e.g., 'Reduce by 20% vs 2020' or 'Not specified'>",
  "energy_savings_obligation_mentioned": "<Strictly 'Yes' or 'No' if an energy savings obligation scheme is mentioned>",
  "building_renovation_rate_target": "<Target for the rate of building renovation, e.g., '3% annual deep renovation rate' or 'Not specified'>",

  "fossil_fuel_subsidy_phaseout_mentioned": "<'Yes', 'No', or specific year/details if phase-out of fossil fuel subsidies is mentioned, e.g., 'Yes, by 2030' or 'No'>",
  "coal_phaseout_year": "<Explicit year mentioned for coal phase-out, e.g., '2038' or 'Not specified'>",
  "ets_integration_details": "<Brief details on integration with EU ETS or national ETS, beyond simple Yes/No, e.g., 'Includes buildings and transport in national ETS' or 'Not specified'>",
  "estimated_investment_needs": "<Estimated total or annual investment needs mentioned for the plan, including currency and period if specified, e.g., 'EUR 50 billion annually until 2030' or 'Not specified'>",
  "key_funding_mechanisms": ["<List of key funding sources or mechanisms mentioned, e.g., 'EU Recovery Fund', 'National climate fund', 'Carbon pricing revenue' or []>"],

  "transport_sector_ghg_target": "<Specific GHG reduction target for the transport sector, e.g., '40% reduction vs 2005' or 'Not specified'>",
  "industry_sector_ghg_target": "<Specific GHG reduction target for the industry sector, e.g., '35% reduction vs 2005' or 'Not specified'>",
  "agriculture_sector_ghg_target": "<Specific GHG reduction target for the agriculture sector, e.g., '20% reduction vs 2005' or 'Not specified'>",
  "ev_adoption_target": "<Target for electric vehicle adoption (number of vehicles or share of sales), e.g., '1 million EVs on the road by 2030' or '50% share of new car sales by 2030' or 'Not specified'>"
}}

--- NECP TEXT START ---
{text_to_summarize}
--- NECP TEXT END ---

Return ONLY the single JSON object...
"""
    # Return the fully constructed prompt string
    return prompt


# Function to call the Gemini model, extract data, and handle retries for rate limits
# MLOps: Incorporates retry logic for API robustness
def extract_data_with_gemini(text_to_summarize, country):
    """Calls Gemini model with retry logic for rate limits."""
    # Build the prompt using the helper function
    # GENAI CAPABILITY: Long context window (Potentially sending large 'text_to_summarize' if files are combined)
    prompt = build_prompt(text_to_summarize, country)
    # Check if the input text is empty; if so, skip the API call
    if not text_to_summarize:
         print(f"Error: Combined text for {country} is empty. Skipping API call.")
         return None

    # Initialize the delay for the first retry attempt
    current_retry_delay = INITIAL_RETRY_DELAY

    # Start the retry loop (includes the initial attempt + MAX_RETRIES)
    for attempt in range(MAX_RETRIES + 1):
        try:
            # Make the API call to the Gemini model to generate content
            # GENAI CAPABILITY: Calling Foundational Model (Gemini API call)
            response = model.generate_content(
                prompt,                         # The detailed prompt with instructions and context
                generation_config=GENERATION_CONFIG, # Generation parameters (temp, max tokens)
                safety_settings=SAFETY_SETTINGS,      # Safety filtering configuration
                # request_options={"timeout": 300} # Optional: Increase network timeout for long requests
            )

            # --- Successful Response Handling ---
            # Check if the response contains any candidates (results)
            if not response.candidates:
                 # If no candidates, check for blocking reasons (safety, etc.)
                 block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                 print(f"Warning: No candidates received from Gemini for {country}. Possibly blocked. Reason: {block_reason}")
                 # Do not retry if blocked by safety or other non-recoverable issues
                 return None

            # Get the first candidate from the response
            candidate = response.candidates[0]

            # Check why the generation finished
            if candidate.finish_reason.name != "STOP":
                # If it didn't stop normally, log a warning
                print(f"Warning: Model generation stopped for {country}. Reason: {candidate.finish_reason.name}")
                # If stopped due to safety or recitation issues, do not retry
                if candidate.finish_reason.name in ["SAFETY", "RECITATION"]:
                     return None
                # Check if content exists even if stopped for other reasons (e.g., MAX_TOKENS)
                if not hasattr(candidate, 'content') or not candidate.content or not candidate.content.parts:
                     print(f"Error: No content parts in response for {country} after stop reason {candidate.finish_reason.name}")
                     # Don't retry if no content available after non-normal stop
                     return None

            # Check again if content parts exist (should exist if finish_reason was STOP or handled above)
            if not hasattr(candidate, 'content') or not candidate.content or not candidate.content.parts:
                 print(f"Error: No content parts found in the response for {country}.")
                 # Don't retry if content is missing unexpectedly
                 return None

            # Extract the raw text content from the first part of the response
            raw_response_text = candidate.content.parts[0].text.strip()

            # Clean potential markdown code fences (```json ... ```) around the response
            if raw_response_text.startswith("```json"): raw_response_text = raw_response_text[7:]
            if raw_response_text.startswith("```"): raw_response_text = raw_response_text[3:]
            if raw_response_text.endswith("```"): raw_response_text = raw_response_text[:-3]
            # Strip leading/trailing whitespace again after cleaning fences
            raw_response_text = raw_response_text.strip()

            # Check if the response is empty after cleaning
            if not raw_response_text:
                print(f"Error: Received empty response text from Gemini for {country} after stripping markdown.")
                # Don't retry if the response was genuinely empty
                return None

            # Attempt to parse the cleaned response text as JSON
            try:
                # GENAI CAPABILITY: Structured Output/JSON Mode (Parsing the LLM's attempt at JSON)
                extracted_data = json.loads(raw_response_text)
                # Basic validation: check if it's a dictionary and has the 'country' key
                if isinstance(extracted_data, dict) and "country" in extracted_data:
                     # Success! Log and return the extracted data dictionary
                     print(f"Success on attempt {attempt + 1} for {country}.")
                     return extracted_data # Exit the retry loop and return data
                else:
                    # Parsed structure isn't the expected dictionary type or lacks the key
                    print(f"Error: Parsed JSON for {country} (Attempt {attempt+1}) is not valid dict or lacks 'country'.")
                    # Don't retry if the fundamental structure is wrong
                    return None
            # Handle errors specifically during JSON decoding
            except json.JSONDecodeError as json_err:
                print(f"Error: Failed to decode JSON response from Gemini for {country} (Attempt {attempt+1}).")
                print(f"JSONDecodeError: {json_err}")
                # Log the problematic text that failed parsing (truncated for brevity)
                print(f"--- Raw Response Text for {country} ---\n{raw_response_text[:500]}...\n--- End Raw Response ---")
                # Don't retry JSON decoding errors, as it likely requires prompt/model changes
                return None
            # Handle any other unexpected errors during the parsing phase
            except Exception as e:
                 print(f"Error: Unexpected error parsing response for {country} (Attempt {attempt+1}): {e}")
                 # Don't retry unknown parsing errors
                 return None
            # --- End response handling block ---

        # --- Exception Handling with Retry Logic ---
        # Specifically catch "ResourceExhausted" errors (HTTP 429 - Rate Limit)
        except google_exceptions.ResourceExhausted as e:
            # Log that the rate limit was hit
            print(f"(!) Rate limit hit for {country} on attempt {attempt + 1}/{MAX_RETRIES + 1}. Error: {e}")
            # Check if the maximum number of retries has been reached
            if attempt >= MAX_RETRIES:
                print(f"ERROR: Max retries ({MAX_RETRIES}) reached for {country}. Giving up.")
                return None # Stop trying after max retries

            # --- Calculate Wait Time ---
            # MLOps: Implementing exponential backoff with optional API-suggested delay
            suggested_delay = None
            # Try to extract the 'retry_delay' value from the error's metadata if available
            try:
                 if e.metadata and hasattr(e.metadata, '__iter__'):
                    for item in e.metadata:
                        # Look for the specific key and value structure
                        if hasattr(item, 'key') and item.key == 'retry_delay' and hasattr(item, 'value'):
                             # Extract the seconds value (assuming format like '41s')
                             suggested_delay = int(item.value.split('s')[0])
                             break # Stop searching once found
            except Exception as meta_e:
                 # Log if parsing the suggested delay fails
                 print(f"    Could not parse suggested retry delay: {meta_e}")

            # Determine the actual wait time
            if suggested_delay is not None and suggested_delay > 0:
                 # Use the suggested delay if it's valid and longer than our current calculated delay
                 wait_time = max(suggested_delay, current_retry_delay)
                 print(f"    API suggested retry delay: {suggested_delay}s.")
            else:
                 # Calculate wait time using exponential backoff formula
                 wait_time = current_retry_delay * (2 ** attempt)
                 # Add random jitter (0 to 1 second) to avoid synchronized retries from multiple processes
                 wait_time += random.uniform(0, 1)
                 # Cap the maximum wait time to avoid excessively long delays
                 wait_time = min(wait_time, MAX_RETRY_DELAY)
                 print(f"    Calculated exponential backoff delay.")

            # Log the wait time and pause execution
            print(f"    Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            # Note: We don't increase current_retry_delay exponentially here, relying more on attempt number or suggested delay

        # Catch other potential Google API errors (like server errors - 5xx)
        except google_exceptions.GoogleAPIError as api_err:
             print(f"(!) Google API Error for {country} on attempt {attempt + 1}: {type(api_err).__name__} - {api_err}")
             # For simplicity, we are not retrying these errors currently, but could be added for specific types (e.g., 503)
             return None
        # Catch any other unexpected exceptions during the API call process
        except Exception as e:
            print(f"ERROR: An unexpected error occurred during API call for {country} (Attempt {attempt+1}): {type(e).__name__} - {e}")
            # Do not retry unknown errors
            return None

    # This line should theoretically not be reached if the loop logic is correct, but acts as a safeguard
    return None # Return None if all retries fail


# Function to orchestrate the processing of text files grouped by country
def process_countries(txt_folder, json_folder):
    """
    Groups txt files by country, combines text, calls extract_data_with_gemini (with retries),
    and saves one JSON file per country.
    """
    # Log the start of the scanning process
    print("Scanning for .txt files and grouping by country...")
    # Use defaultdict to easily create lists of filepaths for each country
    country_files = defaultdict(list)
    txt_files = []
    # Find all .txt files in the input folder, excluding the log file
    try:
         txt_files = [f for f in os.listdir(txt_folder)
                      if f.endswith(".txt") and not f.startswith("extraction_log")]
    except Exception as e: print(f"ERROR listing files: {e}"); return # Handle file listing errors
    # Check if any text files were found
    if not txt_files: print(f"No .txt files found in {txt_folder}. Exiting."); return

    # Iterate through the found text files to group them by country
    for txt_file in txt_files:
        # Construct the full path to the file
        filepath = os.path.join(txt_folder, txt_file)
        # Extract the country name from the file's metadata
        country = get_country_from_metadata(filepath)
        # If a valid country name was extracted, add the filepath to the list for that country
        if country and country != "Unknown Country":
            country_files[country].append(filepath)
        # Log files skipped due to unknown country
        elif country == "Unknown Country":
             print(f"  Skipping file {txt_file} due to unknown country in metadata.")
        # Files skipped due to read errors are logged within get_country_from_metadata

    # Check if any files were successfully grouped
    if not country_files: print("No files grouped by country. Exiting."); return
    # Log the number and names of countries found
    print(f"Found {len(country_files)} countries to process: {', '.join(sorted(country_files.keys()))}")

    # Initialize counter for successfully processed countries
    countries_processed_successfully = 0
    # Iterate through each country and its associated list of filepaths, using tqdm for progress bar
    for country_name, filepaths in tqdm(sorted(country_files.items()), desc="Processing countries"):
        # Log the start of processing for the current country
        print(f"\nProcessing Country: {country_name} (from {len(filepaths)} file(s))")
        # Initialize an empty string to store the combined text for the country
        combined_body_text = ""
        # Sort filepaths alphabetically (helps if files are named like 'part1', 'part2')
        filepaths.sort()
        # Iterate through the sorted filepaths for the current country
        for i, filepath in enumerate(filepaths):
            # Log which file is being read
            print(f"  - Reading: {os.path.basename(filepath)}")
            # Extract the body text (after metadata) from the file
            body = get_body_text(filepath)
            # If text was successfully extracted
            if body:
                # If this is not the first file for the country, add a separator
                if i > 0: combined_body_text += "\n\n===== SEPARATOR FOR NEXT DOCUMENT PART =====\n\n"
                # Append the extracted body text to the combined string
                combined_body_text += body
            # Log a warning if a file yielded empty body text
            else: print(f"    Warning: Got empty body text from {os.path.basename(filepath)}")

        # Check if any text was combined for the country
        if not combined_body_text.strip():
            print(f"Error: No combined text generated for {country_name}. Skipping.")
            continue # Skip to the next country

        # Call the function to extract structured data from the combined text using Gemini (includes retry logic)
        extracted_data = extract_data_with_gemini(combined_body_text, country_name)

        # Sanitize the country name to create a valid filename (replace spaces, etc. with underscores)
        safe_filename_country = re.sub(r'[\\/*?:"<>| ]', '_', country_name)
        # Construct the full path for the output JSON file
        json_path = os.path.join(json_folder, f"{safe_filename_country}.json")

        # Check if the extraction was successful and returned a dictionary
        if extracted_data and isinstance(extracted_data, dict):
             # Verify/correct the country name within the extracted data itself
             if extracted_data.get('country') != country_name:
                  print(f"Warning: LLM returned country '{extracted_data.get('country')}' != expected '{country_name}'. Overwriting.")
                  extracted_data['country'] = country_name # Ensure consistency

             # Try to save the extracted data dictionary to the JSON file
             try:
                 # Open the JSON file in write mode with UTF-8 encoding
                 with open(json_path, "w", encoding="utf-8") as json_file:
                     # Dump the dictionary to the file with indentation for readability, ensuring non-ASCII chars are kept
                     json.dump(extracted_data, json_file, indent=4, ensure_ascii=False)
                 # Log successful saving
                 print(f"Successfully extracted and saved data for {country_name} to {json_path}")
                 # Increment the success counter
                 countries_processed_successfully += 1
             # Handle potential errors during file saving
             except Exception as e: print(f"Error saving JSON for {country_name}: {e}")
        else:
            # Log failure if extraction didn't return valid data (e.g., after max retries)
            print(f"Failed to extract valid data for {country_name} after retries (if applicable). No JSON file saved.")

        # Pause execution before starting the next country to respect API rate limits
        print(f"Waiting {BASE_API_DELAY_SUCCESS}s before next country...")
        time.sleep(BASE_API_DELAY_SUCCESS)

    # Print a summary of the processing results at the end
    print(f"\n--- Processing Summary ---")
    print(f"Attempted processing for {len(country_files)} countries.")
    print(f"Successfully saved JSON summaries for {countries_processed_successfully} countries.")


# --- Main Execution ---
# This block runs only when the script is executed directly
if __name__ == "__main__":
    # Print a starting message
    print("--- Starting Grouped NECP Data Extraction with Retries ---")
    # Call the main processing function
    process_countries(TXT_FOLDER, JSON_FOLDER)
    # Print a completion message
    print("--- Extraction Process Complete ---")

# --- GenAI Capability Analysis Summary ---
# - Document Understanding: LLM processes the concatenated text from NECP files.
# - Structured output/JSON mode/controlled generation: Core functionality via prompt engineering requesting specific JSON structure.
# - Prompt Engineering: Detailed instructions, role-playing, format specification used in `build_prompt`.
# - Long context window: Implicitly leveraged (or potentially exceeded) when sending combined text of multiple files.
# - MLOps (GenAI focus): Implemented robust retry logic with exponential backoff for handling API rate limits (429 errors).
# - Foundational Model Access: Uses `google.generativeai` SDK to call the Gemini model.
# - Content Safety/Filtering: Uses `SAFETY_SETTINGS` parameter in API call.
# - Controlled Generation: Uses `GENERATION_CONFIG` (temperature, max_tokens) to influence output.
# - (Not Used: Few-shot prompting (explicit examples removed/not present in this version's prompt), Image/Video/Audio understanding, Function Calling, Agents, Context caching, Gen AI evaluation, Grounding, Embeddings, RAG, Vector search)