# AI Researcher

A research project focused on AI and machine learning.

## Description

This repository contains research, experiments, and implementations related to artificial intelligence and machine learning, with a focus on using Google's Gemini API.

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

## Usage

[Add usage instructions here]

## Project Structure

```
AI_researcher/
├── README.md
├── .gitignore
└── [your project files]
```

## Setup
export GEMINI_API_KEY="api-key"
export GEMINI_MODEL="models/gemini-2.0-flash-lite"

Run streamlit app: streamlit run viz_run_log_streamlit.py
Run two agent lab: python two_agent_lab_gemini.py --prompt "how to model antibodies more effectively?" --paper paper.txt