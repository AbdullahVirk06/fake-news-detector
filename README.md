🚀 News Analyzer App
A real-time news analysis platform powered by AI.
This project is an AI-powered web application that analyzes news articles and text for sentiment, fake news credibility, and provides a concise AI-generated reasoning for its predictions. It’s built to help users quickly assess the trustworthiness of information they encounter online.

✨ Features
Sentiment Analysis: Determines if a news piece is positive, neutral, or negative.

Fake News Detection: Predicts the likelihood of an article being real or fake news.

AI Reasoning: Uses an advanced LLM to explain why a prediction was made, citing key indicators.

Related Articles: Fetches relevant, verifiable articles from external sources to provide additional context.

Database Integration: Saves a history of all analyzed articles for future reference.

🛠️ Getting Started
To run this project locally, you'll need to set up the necessary API keys.

Prerequisites
Python 3.10 or newer

pip (Python package installer)

Installation
Clone the repository:

clone the repo: https://github.com/AbdullahVirk06/fake-news-detector/settings/access?guidance_task=

Install the required Python packages:

pip install -r requirements.txt

API Keys
This application relies on several external APIs. You must add your API keys as environment variables or secrets for the app to function.

Create a .env file in the root directory and add the following:

SENTIMENT_KEY="your_huggingface_sentiment_token"
FAKENEWS_KEY="your_huggingface_fakenews_token"
GROQ_KEY="your_groq_api_key"
NEWSAPI_KEY="your_newsapi_key"

Note: For deployment platforms like Hugging Face Spaces, you will set these as secret variables in the project settings.

▶️ Running the App
Once you have installed the dependencies and configured your API keys, you can launch the application with a single command:

python app.py

The app will be accessible in your web browser, typically at http://127.0.0.1:7860.

🧠 Technology Stack
Framework: Gradio for the user interface

AI Models:

Hugging Face: Used for sentiment analysis and fake news detection.

Groq: Used for AI reasoning.

Data: NewsAPI for fetching related articles.

Database: SQLite or MongoDB for saving analysis history.

