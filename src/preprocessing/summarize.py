import os
import json
import time
from tqdm import tqdm
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold # For safety settings

# --- Configuration ---
# Set input and output folders
TXT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
JSON_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "summaries")

# --- Google AI Configuration ---
# !! IMPORTANT !! Set your API Key securely as an environment variable
# How to set environment variable:
# Linux/macOS: export GOOGLE_API_KEY="YOUR_API_KEY"
# Windows (cmd): set GOOGLE_API_KEY=YOUR_API_KEY
# Windows (PowerShell): $env:GOOGLE_API_KEY="YOUR_API_KEY"
# Or, replace "YOUR_API_KEY_HERE" below, but this is less secure.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")

# Gemini Model Configuration
# Use a model compatible with the google-generativeai SDK
# Common choices:
# - gemini-1.5-flash-latest : Faster, cost-effective
# - gemini-1.5-pro-latest   : More powerful, potentially better results
# - gemini-1.0-pro          : Older version, still capable
MODEL_NAME = "gemini-1.5-flash-latest" # Or choose another suitable model

# API Call settings (adjust as needed)
GENERATION_CONFIG = {
    "max_output_tokens": 1024, # Increased slightly for potentially longer lists
    "temperature": 0.3,      # Lower temperature for more factual extraction
    "top_p": 0.95,
    # "response_mime_type": "application/json", # Request JSON output directly (Recommended for newer models)
    # Note: Using response_mime_type simplifies parsing but requires the model to reliably output *only* JSON.
    # If the model adds explanations, it might fail. We'll stick to text parsing for robustness initially.
}

# Safety Settings (syntax for google-generativeai)
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# Delay between API calls (in seconds) to avoid hitting rate limits (e.g., 60 QPM free tier limit)
API_DELAY = 5.0 # Adjust based on your usage tier and observed errors (increase if you get 429 errors)

# --- Initialization ---
# Create output folder if it doesn't exist
os.makedirs(JSON_FOLDER, exist_ok=True)

# Configure the Google AI SDK
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
    print("ERROR: GOOGLE_API_KEY is not set.")
    print("Please set the GOOGLE_API_KEY environment variable or replace 'YOUR_API_KEY_HERE' in the script.")
    exit(1)

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Google AI SDK configured successfully.")
except Exception as e:
    print(f"Error configuring Google AI SDK: {e}")
    print("Please ensure your API key is valid.")
    exit(1)

# Load the Gemini model
try:
    model = genai.GenerativeModel(MODEL_NAME)
    print(f"Using Gemini model: {MODEL_NAME}")
except Exception as e:
    print(f"Error loading model '{MODEL_NAME}': {e}")
    print("Check if the model name is correct and available for your API key.")
    exit(1)

# --- Functions ---
def build_prompt(text_to_summarize, country):
    """Constructs the prompt for the Gemini model."""
    # This prompt structure remains the same as it's good practice for few-shot learning
    # Updated instructions slightly for clarity and robustness
    prompt = f"""
You are an expert analyst specializing in National Energy and Climate Plans (NECPs).
Your task is to carefully read the provided NECP text for a specific country and extract the following key information:
- GHG emission reduction target for 2030 compared to a base year (e.g., "55% reduction from 1990 levels"). Be specific about the base year if mentioned. If the target or base year is not clearly stated, return "Not specified".
- Renewable energy share target for 2030 (e.g., "65% share in gross final energy consumption"). Be specific about the scope (electricity, overall consumption) if mentioned. If the target or scope is not clearly stated, return "Not specified".
- Energy efficiency target for 2030 (e.g., "40% reduction in primary energy consumption compared to baseline scenario"). Be specific about primary/final energy and comparison point if mentioned. If the target or details are not clearly stated, return "Not specified".
- Is carbon pricing (like ETS, carbon tax) explicitly mentioned as a policy measure? Answer strictly "Yes" or "No". If unsure or not mentioned, return "No".
- List notable policy measures mentioned to achieve these targets. Focus on concrete actions (e.g., "Phase-out coal by 2038", "Increase electric vehicle charging points", "Carbon tax on transport fuels", "Building renovation strategy", "Support for offshore wind"). Extract up to 5-7 key distinct measures. If no specific measures are found, return an empty list [].

Return ONLY the extracted information in a valid JSON format, adhering strictly to the structure below. Do not include any introductory text, explanations, apologies, markdown formatting (like ```json), or anything else before or after the JSON object itself.

JSON Format:
{{
  "country": "{country}",
  "ghg_target_2030": "<GHG_target_2030 or Not specified>",
  "renewable_target_2030": "<renewable_target_2030 or Not specified>",
  "efficiency_target_2030": "<efficiency_target_2030 or Not specified>",
  "carbon_pricing": "<Yes/No>",
  "measures": ["<measure_1>", "<measure_2>", ...]
}}

--- EXAMPLES ---

Example 1 Input Text (Conceptual Snippet for Germany):
"...Germany commits to a 55% reduction in greenhouse gas emissions by 2030 compared to 1990 levels... The Renewable Energy Sources Act (EEG) aims for a 65% share of renewables in gross electricity consumption... Energy efficiency measures target a 40% decrease in primary energy consumption... The national carbon pricing system for transport and heating fuels complements the EU ETS..."

Example 1 Expected JSON Output:
{{
  "country": "Germany",
  "ghg_target_2030": "55% reduction from 1990 levels",
  "renewable_target_2030": "65% share in gross electricity consumption",
  "efficiency_target_2030": "40% decrease in primary energy consumption",
  "carbon_pricing": "Yes",
  "measures": ["National carbon pricing system (transport/heating)", "Renewable Energy Sources Act (EEG)", "Energy efficiency measures"]
}}

Example 2 Input Text (Conceptual Snippet for France):
"...France's updated goal is a 50% cut in GHG emissions by 2030 (vs 1990)... aims for 40% renewable sources in final energy consumption... drive down final energy consumption by 30%... The existing carbon tax continues to be a key instrument..."

Example 2 Expected JSON Output:
{{
  "country": "France",
  "ghg_target_2030": "50% cut in GHG emissions by 1990",
  "renewable_target_2030": "40% renewable sources in final energy consumption",
  "efficiency_target_2030": "30% reduction in final energy consumption",
  "carbon_pricing": "Yes",
  "measures": ["Carbon tax"]
}}

Example 3 Input Text (Conceptual Snippet for Country X - Limited Info):
"...Country X plans significant investments in wind power. Further details on emission targets will be provided later..."

Example 3 Expected JSON Output:
{{
  "country": "Country X",
  "ghg_target_2030": "Not specified",
  "renewable_target_2030": "Not specified",
  "efficiency_target_2030": "Not specified",
  "carbon_pricing": "No",
  "measures": ["Investments in wind power"]
}}
--- END EXAMPLES ---

Now, analyze the following NECP text for {country}:

--- NECP TEXT START ---
{text_to_summarize}
--- NECP TEXT END ---

Return ONLY the JSON object containing the extracted data for {country}.
"""
    return prompt

def extract_data_with_gemini(text_to_summarize, country):
    """
    Calls the Gemini model via Google AI SDK to extract structured data.

    Args:
        text_to_summarize (str): The content of the NECP file.
        country (str): The name of the country.

    Returns:
        dict: The extracted data as a dictionary, or None if an error occurs
              or the model response is invalid.
    """
    prompt = build_prompt(text_to_summarize, country)

    try:
        # print(f"DEBUG: Sending request for {country}...") # Optional: for debugging
        response = model.generate_content(
            prompt, # Pass the prompt string directly
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS,
            # request_options={"timeout": 120} # Optional: Increase timeout for long requests
        )
        # print(f"DEBUG: Received response for {country}.") # Optional: for debugging

        # --- Response Validation and Parsing ---

        # Check for safety blocks or other issues before accessing text
        if not response.candidates:
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
            print(f"Warning: No candidates received from Gemini for {country}. Possibly blocked. Reason: {block_reason}")
            return None

        # Access the first candidate (usually the only one unless num_candidates is set)
        candidate = response.candidates[0]

        # Check finish reason
        # Compare using the string name instead of the enum value
        if candidate.finish_reason.name != "STOP":
            print(f"Warning: Model generation stopped for {country}. Reason: {candidate.finish_reason.name}")
            # Check safety ratings if stopped due to safety
            # Compare using the string name instead of the enum value
            if candidate.finish_reason.name == "SAFETY":
                for rating in candidate.safety_ratings:
                    if rating.blocked:
                        print(f"  - Blocked due to safety category: {rating.category.name}")
            # If stopped for other reasons (e.g., MAX_TOKENS), content might still exist
            # This part remains the same
            if not candidate.content or not candidate.content.parts:
                print(f"Error: No content parts in response for {country} after stop reason {candidate.finish_reason.name}")
                return None
            # Fall through to try parsing if content exists

        # Extract text content
        if not candidate.content or not candidate.content.parts:
            print(f"Error: No content parts found in the response for {country}, even though finish reason was {candidate.finish_reason.name}.")
            return None

        raw_response_text = candidate.content.parts[0].text.strip() # Use candidate.text

        # Sometimes the model might wrap the JSON in markdown backticks, remove them
        if raw_response_text.startswith("```json"):
            raw_response_text = raw_response_text[7:]
        if raw_response_text.startswith("```"):
             raw_response_text = raw_response_text[3:]
        if raw_response_text.endswith("```"):
            raw_response_text = raw_response_text[:-3]
        raw_response_text = raw_response_text.strip()

        # If the response is empty after stripping, report error
        if not raw_response_text:
            print(f"Error: Received empty response text from Gemini for {country} after stripping markdown.")
            return None

        # Attempt to parse the response text as JSON
        try:
            extracted_data = json.loads(raw_response_text)
            # Basic validation: Check if it's a dictionary and has the expected country key
            if isinstance(extracted_data, dict) and "country" in extracted_data:
                # Optional: Deeper validation (check if keys exist, types are correct)
                # required_keys = ["ghg_target_2030", "renewable_target_2030", "efficiency_target_2030", "carbon_pricing", "measures"]
                # if all(key in extracted_data for key in required_keys) and isinstance(extracted_data.get("measures"), list):
                #      return extracted_data
                # else:
                #      print(f"Error: Parsed JSON for {country} missing required keys or 'measures' is not a list.")
                #      print(f"--- Raw Response Text for {country} ---\n{raw_response_text}\n--- End Raw Response ---")
                #      return None
                return extracted_data # Simpler validation for now
            else:
                print(f"Error: Parsed JSON for {country} is not a valid dictionary or lacks 'country' key.")
                print(f"--- Raw Response Text for {country} ---\n{raw_response_text}\n--- End Raw Response ---")
                return None
        except json.JSONDecodeError as json_err:
            print(f"Error: Failed to decode JSON response from Gemini for {country}.")
            print(f"JSONDecodeError: {json_err}")
            print(f"--- Raw Response Text for {country} ---\n{raw_response_text}\n--- End Raw Response ---")
            return None
        except Exception as e: # Catch other potential errors during parsing
             print(f"Error: Unexpected error parsing response for {country}: {e}")
             print(f"--- Raw Response Text for {country} ---\n{raw_response_text}\n--- End Raw Response ---")
             return None


    except Exception as e:
        # Catch potential API errors (like RateLimitError, APICallError, etc.)
        # The specific exceptions might vary based on the library version
        print(f"Error calling Google AI API for {country}: {type(e).__name__} - {e}")
        # Consider adding specific handling for rate limit errors (e.g., longer sleep)
        if "429" in str(e): # Basic check for rate limit indication
            print("Rate limit likely exceeded. Consider increasing API_DELAY.")
        return None


def process_txt_files(txt_folder, json_folder):
    """
    Processes each .txt file in the input folder, extracts data using Gemini,
    and saves the results as JSON files in the output folder.
    """
    try:
        txt_files = [f for f in os.listdir(txt_folder) if f.endswith(".txt")]
    except FileNotFoundError:
         print(f"ERROR: Input folder '{txt_folder}' not found.")
         return
    except Exception as e:
         print(f"ERROR: Could not list files in input folder '{txt_folder}': {e}")
         return

    if not txt_files:
        print(f"No .txt files found in {txt_folder}. Exiting.")
        return

    print(f"Found {len(txt_files)} .txt files to process.")

    for txt_file in tqdm(txt_files, desc="Processing NECP files"):
        country_name = os.path.splitext(txt_file)[0] # More robust way to remove extension
        txt_path = os.path.join(txt_folder, txt_file)
        json_path = os.path.join(json_folder, f"{country_name}_summary.json")

        # Optional: Skip if JSON already exists
        # if os.path.exists(json_path):
        #     print(f"Skipping {country_name}, JSON already exists.")
        #     time.sleep(0.1) # Small delay even when skipping
        #     continue

        print(f"\nProcessing: {country_name} ({txt_file})")

        try:
            # Read the cleaned text from the file
            with open(txt_path, "r", encoding="utf-8") as file:
                # Consider adding a limit to the text size if files are extremely large
                # text = file.read(MAX_CHARS_PER_FILE) # Example limit
                text = file.read()
                if not text.strip():
                     print(f"Warning: File {txt_file} is empty or contains only whitespace. Skipping.")
                     continue

            # Call the Gemini model to extract data
            extracted_data = extract_data_with_gemini(text, country_name)

            # Check if extraction was successful and data is valid
            if extracted_data and isinstance(extracted_data, dict):
                # Save the extracted data in a JSON file
                try:
                    with open(json_path, "w", encoding="utf-8") as json_file:
                        json.dump(extracted_data, json_file, indent=4, ensure_ascii=False)
                    print(f"Successfully processed and saved summary for {country_name} to {json_path}")
                except IOError as io_err:
                    print(f"Error saving JSON file for {country_name} to {json_path}: {io_err}")
                except Exception as e:
                    print(f"Unexpected error saving JSON for {country_name}: {e}")
            else:
                print(f"Failed to extract valid data for {country_name}. No JSON file saved.")

            # Add a delay to respect potential API rate limits
            # print(f"Waiting {API_DELAY}s before next request...") # Optional feedback
            time.sleep(API_DELAY)

        except FileNotFoundError:
            print(f"Error: File not found {txt_path}. Skipping.")
        except IOError as io_err:
             print(f"Error reading file {txt_path}: {io_err}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred processing {txt_file}: {e}")
            # Optionally add a longer delay or break after unexpected errors
            print(f"Waiting longer delay ({API_DELAY * 2}s) after unexpected error...")
            time.sleep(API_DELAY * 2)


# --- Main Execution ---
if __name__ == "__main__":
    # --- Pre-run Checks ---
    if TXT_FOLDER == "path/to/your/txt_folder" or JSON_FOLDER == "path/to/your/json_folder":
        print("ERROR: Please update TXT_FOLDER and JSON_FOLDER paths in the script.")
        exit(1)
    # API Key check is done during initialization

    print("--- Starting NECP Data Extraction using Google AI SDK ---")
    process_txt_files(TXT_FOLDER, JSON_FOLDER)
    print("--- Processing Complete ---")