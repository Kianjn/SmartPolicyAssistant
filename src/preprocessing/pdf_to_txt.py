# Import necessary libraries
import os                     # For interacting with the operating system (file paths, directories)
import pdfplumber             # For extracting text and tables from PDF files
import re                     # For using regular expressions (pattern matching for text cleaning/extraction)
from datetime import datetime # For getting the current date and time (used in metadata and logs)
import csv                    # Imported but not currently used in this version (kept for potential future use with CSV metadata maps)

# --- Configuration ---
# Define a dictionary mapping country codes (expected as prefixes in PDF filenames) to their full names.
# This map is used to generate more readable country metadata in the output text files.
# NOTE: This requires manual maintenance to ensure codes match filenames and names are correct.
COUNTRY_CODE_MAP = {
    "AT": "Austria",
    "BE": "Belgium",             # Generic code for Belgium
    "BE-FL": "Belgium (Flanders)", # Example for regional codes
    "BE-WAL": "Belgium (Wallonia)",# Example for regional codes
    "BG": "Bulgaria",
    "HR": "Croatia",
    "CY": "Cyprus",
    "CS": "Czech Republic",      
    "DK": "Denmark",
    "EE": "Estonia",
    "FI": "Finland",
    "FR": "France",
    "DE": "Germany",
    "EL": "Greece",              # Common code for Greece
    "GR": "Greece",              # Alternative common code for Greece
    "HU": "Hungary",
    "IE": "Ireland",
    "IT": "Italy",
    "LV": "Latvia",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "MT": "Malta",
    "NL": "Netherlands",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "SK": "Slovakia",
    "SI": "Slovenia",
    "ES": "Spain",
    "SE": "Sweden",
    "EU": "European Union",      # For EU-level documents
    # Add any other relevant country/region codes encountered in filenames
}


# --- Paths ---
# Determine the project's root directory based on the current script's location (assuming src/core/script.py structure)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# Define the path to the folder containing input PDF files
pdf_folder = os.path.join(PROJECT_ROOT, "data", "input")
# Define the path to the folder where processed text files will be saved
txt_folder = os.path.join(PROJECT_ROOT, "data", "processed")
# Define the full path for the extraction log file within the processed folder
log_path = os.path.join(txt_folder, "extraction_log.txt")
# Create the output text folder if it doesn't already exist; exist_ok=True prevents errors if it does exist
os.makedirs(txt_folder, exist_ok=True)

# --- Helper Functions ---

# Function to perform basic text cleaning using regular expressions
def clean_text(text):
    # Replace multiple consecutive newline characters with a single newline
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple consecutive spaces or tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove spaces immediately followed by a newline
    text = re.sub(r' +\n', '\n', text)
    # Remove spaces immediately preceding a newline
    text = re.sub(r'\n +', '\n', text)
    # Remove leading/trailing whitespace from the entire text
    return text.strip()

# Function to extract a potential country code from the beginning of a filename
def extract_country_code_from_filename(filename):
    """
    Extracts the likely country code from the beginning of the filename.
    Converts to uppercase for consistent map lookup.
    Adjust regex if your filenames have a different pattern.
    """
    # Try to match alphabetical characters and hyphens at the start of the filename
    # This assumes the code precedes separators like '_', '-', '.', or digits.
    match = re.match(r"^([a-zA-Z\-]+)", filename)
    if match:
        # If a match is found, return the captured group (the code) converted to uppercase
        return match.group(1).upper()

    # Fallback strategy: If the regex doesn't match, split the filename by common delimiters
    parts = re.split(r"[_\-\.]", filename)
    if parts:
        # Assume the first part is the code
        potential_code = parts[0].upper()
        # Basic check: ensure the potential code has a plausible length (1 to 5 chars)
        if 1 <= len(potential_code) <= 5:
             return potential_code # Return the uppercase potential code

    # If no plausible code is found through either method, return "UNKNOWN"
    return "UNKNOWN"

# --- Main Processing ---
# Initialize counters for successful and failed PDF extractions
success_count = 0
fail_count = 0
# Initialize a list to store log messages, starting with a timestamped header
log_lines = [f"Extraction Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]

# Get a list of all files in the pdf_folder that end with ".pdf" and are actual files (not directories)
pdf_files_in_folder = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf") and os.path.isfile(os.path.join(pdf_folder, f))]

# Loop through each identified PDF filename
for filename in pdf_files_in_folder:
    # Construct the full path to the current PDF file
    pdf_path = os.path.join(pdf_folder, filename)
    # Create the corresponding output filename by replacing the ".pdf" extension with ".txt"
    txt_filename = os.path.splitext(filename)[0] + ".txt"
    # Construct the full path for the output text file
    txt_path = os.path.join(txt_folder, txt_filename)

    # --- Determine Country Name ---
    # Extract the country code from the current PDF filename
    extracted_code = extract_country_code_from_filename(filename)
    # Look up the full country name in the COUNTRY_CODE_MAP using the extracted code.
    # If the code is not found in the map, use a default string indicating the unknown code.
    country_name = COUNTRY_CODE_MAP.get(extracted_code, f"Unknown Country ({extracted_code})")
    # Set the initial metadata source information
    metadata_source_info = f"Mapped from code '{extracted_code}' in filename"

    # Check if the country name lookup failed (i.e., used the default fallback)
    if country_name.startswith("Unknown Country"):
        # Print a warning to the console
        print(f"  -> Warning: Code '{extracted_code}' from '{filename}' not in COUNTRY_CODE_MAP. Using fallback.")
        # Add a warning message to the log file
        log_lines.append(f"[!] WARNING: Country code '{extracted_code}' not found in map for {filename}.")
        # Update the metadata source info to reflect the fallback
        metadata_source_info = f"Code '{extracted_code}' from filename (not in map)"
    else:
        # If lookup was successful, print the mapping to the console
        print(f"  -> Mapped code '{extracted_code}' to Country='{country_name}' for {filename}")

    # --- PDF Extraction Logic ---
    # Use a try-except block to handle potential errors during PDF processing
    try:
        # Open the PDF file using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            # Initialize an empty string to accumulate all extracted text from the PDF
            all_text = ""
            # Iterate through each page in the PDF
            # GENAI CAPABILITY: Document Understanding (Basic - via pdfplumber library)
            # pdfplumber analyzes the PDF structure to extract text and table elements.
            # Note: This is library-based, not typically what's meant by advanced GenAI document understanding (e.g., using multimodal LLMs).
            for page_num, page in enumerate(pdf.pages):
                # Extract tables from the current page
                tables = page.extract_tables()
                # Initialize an empty string to store text extracted from tables on this page
                table_text = ""
                # Check if any tables were found
                if tables:
                    # Iterate through each table found on the page
                    for table in tables:
                        # Check if the table object is valid
                        if table:
                             # Iterate through each row in the table
                             for row in table:
                                 # Check if the row object is valid
                                 if row:
                                     # Join the cells in the row with tabs. Convert cells to strings, handle None values.
                                     row_text = "\t".join([str(cell) if cell is not None else "" for cell in row])
                                     # Append the row text and a newline to the table_text accumulator
                                     table_text += row_text + "\n"
                        # Add an extra newline to separate tables visually in the output text
                        table_text += "\n"

                # Extract the main body text from the page, using tolerances for character spacing.
                # Use empty string "" as fallback if no text is extracted.
                body_text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
                # Combine the body text and the extracted table text
                combined = body_text + "\n" + table_text
                # Clean the combined text using the helper function
                cleaned = clean_text(combined)
                # Append the cleaned text from this page and two newlines (page separator) to the overall text
                all_text += cleaned + "\n\n"

        # Perform a final cleaning pass on the accumulated text from all pages
        all_text = clean_text(all_text)

        # --- Construct Metadata Block ---
        # Create a multi-line f-string containing metadata about the extraction process.
        metadata = f"""=== METADATA ===
Filename: {filename}
Country: {country_name}
Date of Extraction: {datetime.now().strftime('%Y-%m-%d')}
Tool: pdfplumber
Metadata Source: {metadata_source_info}
Notes: Auto-cleaned text + table content, figures omitted.
==================

"""
        # --- Write Output File ---
        # Open the target text file in write mode with UTF-8 encoding
        with open(txt_path, "w", encoding="utf-8") as f:
            # Write the metadata block followed by the extracted and cleaned text
            f.write(metadata + all_text)

        # Log the successful extraction, including the country name
        log_lines.append(f"[✓] SUCCESS: {filename} extracted to {txt_filename} (Country: {country_name})")
        # Increment the success counter
        success_count += 1

    # Handle any exceptions that occur during the PDF processing for this file
    except Exception as e:
        # Log the error, including the filename and the error message
        log_lines.append(f"[!] ERROR: {filename} failed — {str(e)}")
        # Increment the failure counter
        fail_count += 1

# --- Logging Summary ---
# Append a summary section separator to the log lines
log_lines.append(f"\n--- SUMMARY ---")
# Append the total number of PDFs processed
log_lines.append(f"Total PDFs processed: {success_count + fail_count}")
# Append the count of successful extractions
log_lines.append(f"Successful: {success_count}")
# Append the count of failed extractions
log_lines.append(f"Failed: {fail_count}")

# Open the log file in write mode with UTF-8 encoding
with open(log_path, "w", encoding="utf-8") as log_file:
    # Write all accumulated log lines, joined by newline characters
    log_file.write("\n".join(log_lines))

# Print a final completion message to the console, indicating where the log file is saved
print(f"\n[✓] Extraction complete. Log saved to: {log_path}")

# --- GenAI Capability Analysis ---
# This script primarily uses the `pdfplumber` library for PDF text/table extraction and
# standard Python features (regex, file I/O, dictionaries).
# While `pdfplumber` performs basic document understanding, this script does NOT utilize
# any of the listed advanced GenAI capabilities such as LLMs, embeddings, vector stores, etc.