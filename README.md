# AI-Powered Food Recommendation System

## Project Overview

This project is a machine learning-based food recommendation system. It uses TF-IDF vectorization and cosine similarity to suggest food items based on user preferences. Additionally, it features sentiment analysis to refine recommendations based on user mood.

## How It Works

The system takes user input regarding food preferences and mood. It then processes the input using Natural Language Processing (NLP) techniques to find the best-matching food items from a predefined dataset. If the user's mood is negative, the system introduces more diverse recommendations to improve the experience.

## Machine Learning Approach

TF-IDF Vectorization: Converts textual food descriptions into numerical vectors.

Cosine Similarity: Measures how similar the user input is to existing food descriptions.

Sentiment Analysis: Determines the user's mood and adjusts recommendations accordingly.

## Features

✅ Personalized food recommendations using ML✅ Sentiment analysis to adjust recommendations✅ Flask API for real-world usage✅ Logging system to track recommendations✅ Endpoints for fetching available foods & descriptions

Project Structure

📂 AI-Food-Recommendation  
├── 📜 README.md  
├── 📜 requirements.txt  
├── 📜 app.py  
├── 📂 data  
│   ├── 📜 dataset.csv  
├── 📂 logs  
│   ├── 📜 recommendation_logs.json  

## Installation

1️⃣ Clone the repository:

git clone https://github.com/yourusername
cd AI-Food-Recommendation

2️⃣ Install dependencies:

pip install -r requirements.txt

3️⃣ Run the API:

python app.py

API Endpoints

1️⃣ Get Food Recommendations

Endpoint: /recommend (POST)Request:

{
  "preference": "I love spicy and flavorful dishes",
  "mood": "happy"
}

Response:

{
  "recommendations": ["Curry", "Tacos", "Pasta"],
  "mood": "positive"
}

2️⃣ Analyze Sentiment

Endpoint: /sentiment (POST)Request:

{
  "text": "I love delicious food!"
}

Response:

{
  "sentiment": "positive"
}

3️⃣ Get All Available Foods

Endpoint: /foods (GET)Response:

{
  "available_foods": ["Pizza", "Burger", "Pasta", "Sushi", "Tacos"]
}

4️⃣ Get Food Description

Endpoint: /food-info (GET)Request:

{
  "name": "Pizza"
}

Response:

{
  "food": "Pizza",
  "description": "Cheesy and delicious Italian dish with tomato sauce"
}

## Future Enhancements

Expand the dataset to include more diverse food items.

Improve sentiment analysis using deep learning models.

User authentication to provide personalized history-based recommendations.
