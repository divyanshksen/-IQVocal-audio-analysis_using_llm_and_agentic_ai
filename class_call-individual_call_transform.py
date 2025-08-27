import os
from dotenv import load_dotenv
from individual_call_transform import OpenRouterTranscriptSummarizer  
import argparse

# 1. Load .env environment variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if api_key is None:
    raise ValueError("❌ API key not found. Please set OPENROUTER_API_KEY in your .env file.")

# 2. Create Summarizer object
summarizer = OpenRouterTranscriptSummarizer(api_key=api_key)

# 3. Set up a Command Line Interface (CLI) to pass arguments when running your script from the terminal
parser = argparse.ArgumentParser(description="Summarize a customer support call transcript.")
parser.add_argument('--csv', required=True, help="Path to the input CSV file.")
parser.add_argument('--call_id', default="call_001", help="Call ID (optional, default: call_001).")
parser.add_argument('--representative', default="Agent_Unknown", help="Representative name (optional).")

args = parser.parse_args()

# 4. Run `summarize_from_csv` function from `individual_call_transform.py`
print(f"Summarizing call: {args.call_id}...")
summarizer.summarize_from_csv(
    csv_input=args.csv,
    call_id=args.call_id,
    representative=args.representative
)
print(f"Summary complete! Reports saved under './reports/html/'.")
