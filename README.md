# -IQVocal-analysis_using_llm_and_agentic_ai

## Project Overview
In today's contact centers, thousands of customer service calls occur daily—each rich with insights but trapped in unstructured audio.
VoiceIQ unlocks this hidden intelligence by transforming raw call data into structured, lightweight, and actionable insights.

Our platform performs:

Speaker-separated transcription and diarization
Sentiment analysis, intent detection, keyword extraction
Contextual call summarization using open-source LLMs
Structured HTML report generation for each call
Agentic Retrieval-Augmented Generation (RAG) chatbot to explore call data
An interactive Streamlit dashboard summarizing agent performance and customer sentiment trends
Beyond intelligence, VoiceIQ optimizes storage by replacing heavy audio files with structured, metadata-rich documents—reducing storage needs by over 99%.

Note:
This project repository is created in partial fulfillment of the requirements for the Big Data Analytics course offered by the Master of Science in Business Analytics program at the Carlson School of Management, University of Minnesota.

## Key Components
Module	Description
individual_call_transform.py	Transcribes and processes individual call audio files, applying speaker diarization and summarization
class_call-individual_call_transform.py	Class-based modular version of call processing for reusability
streamlit_dashboard.py	Streamlit app for visualizing agent performance, call intents, and customer sentiment trends
HTML Call Reports	Auto-generated structured reports per call
RAG Chatbot (Demonstration)	LLM-powered chatbot to answer user queries about historical call trends

## Setup and Installation
1. Clone the repository.
git clone https://github.com/rachellg99/VoiceIQ-llm-driven-audio-analysis.git
cd voiceiq-audio-analyzer
2. Create a virtual environment (recommended):
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install requirements:
pip install -r requirements.txt
4. Install ffmpeg (needed for audio processing): On mac:
brew install ffmpeg
On Ubuntu:

sudo apt-get install ffmpeg
5. Set up Hugging Face Authentication (for Whisper and Pyannote) and OpenRouter API keys:
- Create a Hugging Face token
- Authenticate in code or via huggingface-cli login.
Update the Python code to use your keys.

## How to Run
1. Generate Call Insights:
python class_call-individual_call_transform.py
This will create structured JSON and HTML reports under /reports/.

3. RAGBot Demonstration: Use the metadata from /reports/json/ to answer natural language queries through a local RAG pipeline.

  ## Sample Visuals
Individual call summaries in HTML
Intent distribution charts
Customer sentiment breakdowns by agent and intent
Natural language queries against call database (via RAGBot)
