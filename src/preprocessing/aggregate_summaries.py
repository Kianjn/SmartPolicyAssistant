# --- aggregate_summaries.py (Simple Aggregator) ---

# Import necessary standard Python libraries
import os                         # For interacting with the operating system (paths, directories)
import json                       # For loading data from JSON files

# Import data manipulation library
import pandas as pd               # Used for creating and managing DataFrames (tabular data structures)

# Import progress bar library
from tqdm import tqdm             # Used for displaying a progress bar during file processing loop

# --- Configuration ---
# Determine the project's root directory based on the current script's location
# Assumes a structure like project_root/src/core/script.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define the path to the directory where the individual JSON summary files (output of summarize.py) are stored
JSON_FOLDER = os.path.join(PROJECT_ROOT, "data", "processed", "summaries")

# Define the path and filename for the final aggregated CSV output file
# This file will contain the combined data from all individual JSON files.
AGGREGATED_OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "aggregated_necp_data.csv")


# Define the main function to aggregate JSON files
def aggregate_json_files(json_dir, output_csv):
    """
    Reads all JSON files in a directory, aggregates their data into a single
    Pandas DataFrame, and saves the result as a CSV file.

    Args:
        json_dir (str): Path to the directory containing individual JSON files.
        output_csv (str): Path where the aggregated CSV file should be saved.
    """
    # Initialize an empty list to store the data loaded from each JSON file
    all_data = []
    # Initialize an empty list to store the names of the JSON files found
    json_files = []

    # Print a message indicating the directory being scanned
    print(f"Looking for JSON files in: {json_dir}")
    # Use a try-except block to handle potential errors during file listing
    try:
        # Get a list of all files in the specified directory that end with ".json"
        json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    # Handle the case where the input directory doesn't exist
    except FileNotFoundError:
        print(f"ERROR: Input directory not found: {json_dir}")
        return # Exit the function if the directory is missing
    # Handle any other errors that might occur during directory listing
    except Exception as e:
        print(f"ERROR: Could not list files in directory {json_dir}: {e}")
        return # Exit the function on other listing errors

    # Check if any JSON files were found in the directory
    if not json_files:
        print("No JSON files found to aggregate.")
        return # Exit if no JSON files are present

    # Print the number of JSON files found
    print(f"Found {len(json_files)} JSON files. Reading and aggregating...")

    # Loop through each found JSON filename, using tqdm to show a progress bar
    for filename in tqdm(json_files, desc="Aggregating JSON files"):
        # Construct the full path to the current JSON file
        filepath = os.path.join(json_dir, filename)
        # Use a try-except block to handle errors during file reading or JSON parsing for individual files
        try:
            # Open the JSON file in read mode with UTF-8 encoding
            with open(filepath, 'r', encoding='utf-8') as f:
                # Load the JSON data from the file into a Python object (expected to be a dictionary)
                data = json.load(f)
                # Check if the loaded data is a dictionary (as expected from summarize.py)
                if isinstance(data, dict):
                    # If it's a dictionary, append it to the all_data list
                    # Optionally, you could add the source filename to the dictionary here for traceability:
                    # data['source_json_file'] = filename
                    all_data.append(data)
                else:
                    # Print a warning if the file content is not a dictionary
                    print(f"Warning: Skipping {filename} as its content is not a dictionary (JSON object).")
        # Handle errors if the file contains invalid JSON
        except json.JSONDecodeError:
            print(f"Warning: Skipping {filename} due to JSON decoding error.")
        # Handle any other unexpected errors during file loading/processing
        except Exception as e:
            print(f"Warning: Skipping {filename} due to unexpected error during loading: {e}")

    # After processing all files, check if any valid data was loaded
    if not all_data:
        print("No valid data loaded from JSON files. Cannot create aggregated file.")
        return # Exit if no data was successfully loaded

    # Print a message indicating the creation of the Pandas DataFrame
    print("Creating Pandas DataFrame...")
    # Create a Pandas DataFrame from the list of dictionaries.
    # Pandas automatically handles cases where dictionaries have different keys;
    # missing keys in a dictionary will result in NaN (Not a Number) values in the corresponding DataFrame cell.
    df = pd.DataFrame(all_data)

    # Print the dimensions (rows, columns) of the created DataFrame
    print(f"DataFrame created with {len(df)} rows and {len(df.columns)} columns.")
    # Optional: Uncomment the next line to print the list of column names for inspection
    # print(df.columns.tolist())

    # --- Save to CSV ---
    # Use a try-except block to handle potential errors during CSV writing
    try:
        # Ensure the directory for the output CSV file exists; create it if necessary
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        # Save the DataFrame to a CSV file.
        # index=False prevents Pandas from writing the DataFrame index as a column in the CSV.
        # encoding='utf-8' ensures compatibility with various characters.
        df.to_csv(output_csv, index=False, encoding='utf-8')
        # Print a success message indicating the output file path
        print(f"Successfully saved aggregated data to: {output_csv}")
    # Handle any errors that occur during the CSV saving process
    except Exception as e:
        print(f"ERROR: Failed to save aggregated data to CSV ({output_csv}): {e}")


# --- Main Execution ---
# This block executes only when the script is run directly (not imported as a module)
if __name__ == "__main__":
    # Print a starting message for the aggregation process
    print("--- Starting Aggregation of NECP JSON Summaries ---")
    # Call the main aggregation function with the configured input and output paths
    aggregate_json_files(JSON_FOLDER, AGGREGATED_OUTPUT_CSV)
    # Print a completion message
    print("--- Aggregation Complete ---")

# --- GenAI Capability Analysis ---
# This script performs data aggregation using standard Python libraries (os, json) and
# the Pandas library for data manipulation.
# It does NOT utilize any specific GenAI capabilities listed previously (LLMs, embeddings, etc.).
# Its role is purely data transformation based on the structured output generated by summarize.py.