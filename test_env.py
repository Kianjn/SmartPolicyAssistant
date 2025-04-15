import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Print the current working directory
print(f"Current working directory: {os.getcwd()}")
print(f"GOOGLE_API_KEY exists: {os.getenv('GOOGLE_API_KEY') is not None}")
print(f"GOOGLE_API_KEY value: {os.getenv('GOOGLE_API_KEY')}") 