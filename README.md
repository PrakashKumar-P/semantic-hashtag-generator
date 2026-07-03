# Semantic Hashtag Generator

An NLP-based recommendation system that generates context-aware hashtags from social media captions, built with an interactive Streamlit web app.

## Overview

This project analyzes the semantic content of a social media caption and recommends relevant, context-aware hashtags — going beyond simple keyword matching to understand meaning and intent.

## Results

- **Top-5 Recommendation Accuracy:** 95.1% (evaluated during system testing)

## Tech Stack

- **Language:** Python
- **NLP:** Text preprocessing, semantic similarity techniques
- **Web Framework:** Streamlit
- **ML:** Scikit-learn / NLP libraries used for embedding & similarity scoring

## Features

- Caption input with real-time hashtag generation
- Semantic similarity matching to recommend contextually relevant hashtags (not just keyword extraction)
- Interactive Streamlit interface for instant testing and iteration
- Evaluation pipeline measuring Top-5 recommendation accuracy

## How It Works

1. Input caption is cleaned and preprocessed (tokenization, normalization)
2. Semantic embeddings are generated to capture the meaning of the caption
3. Candidate hashtags are ranked by semantic similarity to the caption
4. Top recommendations are returned and displayed in the Streamlit interface

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/semantic-hashtag-generator.git
cd semantic-hashtag-generator

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

Then open the local URL Streamlit provides (usually `http://localhost:8501`).

## Project Structure

```
semantic-hashtag-generator/
├── app.py                 # Streamlit application entry point
├── src/                    # Core NLP/preprocessing logic
├── notebooks/               # Experimentation & evaluation
├── requirements.txt
└── README.md
```

## Future Improvements

- Support for multi-language captions
- Trending hashtag integration (real-time relevance)
- Fine-tune embeddings on a larger, domain-specific dataset

## Author

**Prakash Kumar P**
[LinkedIn](https://linkedin.com/in/prakash-kumar-p-61a321413) · prakashkumar.20it@gmail.com
