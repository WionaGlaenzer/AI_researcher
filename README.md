# AI Researcher

A multi-agent system to develop hypotheses.

## Description

The AI-researcher uses three agents:
- The Innovator
- The Critic
- The Synthesizer

They are connected via LangGraph and have access to tools such as the ArXiv API to gather more information.

## Getting Started

### Prerequisites

- Python 3.8+
- A Google API key for Gemini (get one at [Google AI Studio](https://makersuite.google.com/app/apikey))

### Installation

```bash
# Clone the repository
git clone https://github.com/WionaGlaenzer/AI_researcher.git
cd AI_researcher

# Install dependencies
pip install -r requirements.txt
```

## Setup
```bash
pip install -U langgraph google-generativeai numpy arxiv pdfminer.six requests certifi
export GEMINI_API_KEY="api-key"
export GEMINI_MODEL="models/gemini-2.0-flash-lite"
# optional:
export GEMINI_EMBED_MODEL="models/text-embedding-004"
export MAX_TOTAL_CHUNKS=3000
export MAX_CHARS_PER_FILE=300000
```

## Usage

### Run two agent lab:

```bash
python two_agent_lab_gemini.py --prompt "how to model antibodies more effectively?" --paper paper.txt
```

### Run streamlit app:

```bash
streamlit run viz_run_log_streamlit.py
```

## Project Structure

```
AI_researcher/
├── README.md
├── .gitignore
├── requirements.txt
└── scr/
    └── visualization/
        └── viz_run_log_streamlit.py
    └── two_agent_lab_gemini.py
└── data/
    ├── paper.txt
    └── arxiv_output/
└── output/
    └── run_log.json

```