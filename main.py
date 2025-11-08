import requests
import json
import time
import os
from datetime import datetime

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

# IMPORTANT: If running this outside a platform like Canvas, replace the empty 
# string with your actual Gemini API key.
API_KEY = "AIzaSyDyWeyzxV8TcuXyKt7h1kMyc4LwPTOw1co" 

# The model and endpoint to use for grounded generation
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

# File paths
UNIVERSITY_JSON_FILENAME = "tester.json" # File to load data from
COLLEGES_DIR = "colleges"  # Directory to store individual college files
POLITENESS_DELAY_SECONDS = 3 # Required delay between API calls

# Ensure the colleges directory exists
os.makedirs(COLLEGES_DIR, exist_ok=True)

def sanitize_filename(filename):
    """Convert a string into a safe filename."""
    # Replace invalid filename characters with underscores
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename.strip()

# -----------------------------------------------------------------------------
# Data Loading and LLM Generation Functions
# -----------------------------------------------------------------------------

def load_university_data(filepath):
    """Loads university data from a JSON file."""
    if not os.path.exists(filepath):
        print(f"FATAL: University data file not found at: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"FATAL: Error loading or parsing JSON file {filepath}: {e}")
        return None

def generate_transfer_data(university_name, domain):
    """
    Constructs a request to the Gemini API to search the web for transfer 
    requirements and output a minimalist, delimited text string.
    """
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    system_prompt = f"""
    You are an expert data extraction agent. Your task is to perform a grounded web search 
    focused ONLY on finding comprehensive transfer student **admissions requirements and application components** for {university_name} ({domain}). Exclude all financial aid and cost information.

    Your goal is to extract all information a prospective transfer student needs to successfully apply, 
    organized into three key areas: General Admissions, Credit Transfer, and Major-Specific requirements.

    You MUST output a single, RAW TEXT string with the following delimiters. 
    Do NOT include any introductory or concluding text, Markdown formatting, or comments.

    1. Output general admissions requirements and application components under the tag: --- GENERAL_INFO_START ---
    - Include **Minimum GPA** (by college/major if specified), **Application Deadlines** (priority/final), and **Required Tests** (e.g., SAT/ACT for exceptions).
    - Detail **Transfer Pathways** accepted (e.g., TAG, Associate Degree for Transfer (ADT), articulation agreements).
    - Detail all **Application Components Required**: List the number of required **Essays/Personal Insight Questions (PIQs)**, whether **Letters of Recommendation (LORs)** are accepted/required, and if a **Portfolio** or **Interview** is part of the process.
    - Detail **General Education Certification** accepted (e.g., IGETC, CSU Breadth), **Maximum Transferable Units**, and **Residency Requirements** (how many units must be taken at the university).

    2. For **EVERY MAJOR** you encounter on the transfer admissions pages, start a new block with the tag: --- MAJOR_START ---
    - **Crucial Instruction for Breadth:** If specific lower-division course numbers are not listed directly on the transfer page, you MUST still list the major and report the *highest-level guidance* available (e.g., "Requires completion of all IGETC/GE" or "Highly competitive, refer to ASSIST.org for specific course sequence.").
    - If specific course prerequisites ARE found, inside this block, provide the **Major Name**, a list of **Required Lower-Division Courses** (course names/numbers), **Minimum Grade** requirements for those courses, and any **Major Selectivity** status (e.g., "Impacted," "Highly Competitive," enrollment restrictions).

    This final output must be machine-readable and concise.
    """
    
    user_query = f"""
    Find all major-specific and general transfer prerequisites for {university_name} using the search domain {domain}. 
    Output the data in the requested raw text, delimited format.
    """

    # Construct the API Payload
    payload = {
        "contents": [{ "parts": [{ "text": user_query }] }],
        "tools": [{ "google_search": {} }],
        "systemInstruction": {
            "parts": [{ "text": system_prompt }]
        },
    }

    # Make the API Call with exponential backoff
    max_retries = 5
    initial_delay = 1
    
    for attempt in range(max_retries):
        try:
            if not API_KEY:
                print("FATAL: API_KEY is missing. Skipping LLM call.")
                return None
                
            response = requests.post(
                API_URL, 
                headers={'Content-Type': 'application/json'}, 
                data=json.dumps(payload),
                timeout=30 
            )
            response.raise_for_status() 

            result = response.json()
            report_text = result.get('candidates', [{}])[0] \
                              .get('content', {}) \
                              .get('parts', [{}])[0] \
                              .get('text', 'Failed to generate report text.')
                            
            # Prepend metadata to the LLM's raw output
            header = f"""
--- UNIVERSITY_START ---
NAME: {university_name}
DOMAIN: {domain}
REPORT_DATE: {current_time}
"""
            return header.strip() + "\n" + report_text + "\n"

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"Request failed for {domain} (Attempt {attempt + 1}/{max_retries}). Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"Failed to generate data for {domain} after {max_retries} attempts. Error: {e}")
                return f"--- UNIVERSITY_START ---\nNAME: {university_name}\nDOMAIN: {domain}\nREPORT_DATE: {current_time}\nSTATUS: FAILED - {e}\n"
        except Exception as e:
            print(f"An unexpected error occurred for {domain}: {e}")
            return f"--- UNIVERSITY_START ---\nNAME: {university_name}\nDOMAIN: {domain}\nREPORT_DATE: {current_time}\nSTATUS: FAILED - {e}\n"
    
    return None

# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    
    CALIFORNIA_UNIVERSITIES = load_university_data(UNIVERSITY_JSON_FILENAME)
    
    if not CALIFORNIA_UNIVERSITIES:
        print("Script stopped due to missing or invalid university data.")
        # If the script is run in a separate environment, we need to ensure the JSON file is present.
        print(f"Please ensure '{UNIVERSITY_JSON_FILENAME}' is in the same directory.")
        exit()

    print(f"Starting database generation for {len(CALIFORNIA_UNIVERSITIES)} California universities...")
    
    total_processed = 0
    
    for uni in CALIFORNIA_UNIVERSITIES:
        name = uni['college_name']
        # Use the domain from the URL
        domain = uni['url'].split('//')[1].split('/')[0]
        
        print(f"\n[{total_processed + 1}/{len(CALIFORNIA_UNIVERSITIES)}] Processing {name} ({domain})...")
        
        # Generate the raw data for the university
        data_block = generate_transfer_data(name, domain)

        if data_block:
            # Create a safe filename for this university
            safe_filename = sanitize_filename(name)
            file_path = os.path.join(COLLEGES_DIR, f"{safe_filename}.txt")
            
            # Check if file exists and handle appropriately
            if os.path.exists(file_path):
                print(f"⚠️  Warning: File already exists for {name}, overwriting...")
            
            # Save the data to an individual file
            try:
                # Open in write mode, which will overwrite any existing file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(data_block)
                print(f"✅ Data {'updated' if os.path.exists(file_path) else 'saved'} to {file_path}")
            except PermissionError as e:
                print(f"❌ Error: No permission to write to {file_path}: {e}")
            except IOError as e:
                print(f"❌ Error saving data for {name}: {e}")
            except Exception as e:
                print(f"❌ Unexpected error while saving data for {name}: {e}")
            
        total_processed += 1
        
        # Mandated politeness delay
        if total_processed < len(CALIFORNIA_UNIVERSITIES):
            print(f"Pausing for {POLITENESS_DELAY_SECONDS} seconds...")
            time.sleep(POLITENESS_DELAY_SECONDS) 

    print("\n" + "="*80)
    print(f"✅ Database Generation Complete! Total universities processed: {total_processed}")
    print(f"Output saved to {COLLEGES_DIR}/ directory")
    print("="*80)