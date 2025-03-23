import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import flask
from flask import Flask, request, jsonify
import random
import logging
import json

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Sample dataset of food items with descriptions
data = {
    "food": ["Pizza", "Burger", "Pasta", "Sushi", "Tacos", "Salad", "Steak", "Ramen", "Curry", "Ice Cream"],
    "description": [
        "Cheesy and delicious Italian dish with tomato sauce",
        "Juicy beef patty with lettuce and tomato",
        "Creamy Alfredo or tangy tomato-based Italian pasta",
        "Japanese rice rolls with fresh fish and seaweed",
        "Mexican dish with seasoned meat and fresh toppings",
        "Healthy bowl of greens, veggies, and dressing",
        "Grilled to perfection with garlic butter",
        "Japanese noodle soup with savory broth",
        "Spicy and flavorful Indian dish with rice",
        "Sweet frozen dessert with various flavors"
    ]
}

# Convert dataset into DataFrame
df = pd.DataFrame(data)

# TF-IDF Vectorizer for similarity calculation
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["description"])

# Function to get food recommendations
def recommend_food(preference, num_recommendations=3):
    input_vec = vectorizer.transform([preference])
    similarities = cosine_similarity(input_vec, X).flatten()
    recommended_indices = similarities.argsort()[-num_recommendations:][::-1]
    recommendations = df.iloc[recommended_indices]["food"].tolist()
    return recommendations

# Sentiment Analysis Mock Function (for mood-based refinement)
def analyze_sentiment(text):
    positive_words = ["love", "great", "happy", "delicious", "tasty", "awesome"]
    negative_words = ["hate", "bad", "sad", "awful", "disgusting", "boring"]
    score = sum(1 for word in text.lower().split() if word in positive_words) - sum(1 for word in text.lower().split() if word in negative_words)
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"

# Function to save logs
def save_log(data):
    with open("recommendation_logs.json", "a") as log_file:
        log_file.write(json.dumps(data) + "\n")

# Flask API Endpoint for Recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    preference = data.get("preference", "")
    mood = data.get("mood", "neutral")
    
    if not preference:
        return jsonify({"error": "No preference provided"}), 400
    
    recommendations = recommend_food(preference)
    sentiment = analyze_sentiment(mood)
    
    if sentiment == "negative":
        recommendations = random.sample(df["food"].tolist(), 3)
    
    response = {"recommendations": recommendations, "mood": sentiment}
    save_log(response)
    return jsonify(response)

# Flask API Endpoint for Sentiment Analysis
@app.route('/sentiment', methods=['POST'])
def sentiment():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    sentiment_result = analyze_sentiment(text)
    return jsonify({"sentiment": sentiment_result})

# Additional Feature: Get all available foods
@app.route('/foods', methods=['GET'])
def get_foods():
    return jsonify({"available_foods": df["food"].tolist()})

# Additional Feature: Get food description
@app.route('/food-info', methods=['GET'])
def get_food_info():
    food_name = request.args.get("name", "").capitalize()
    if food_name in df["food"].values:
        description = df[df["food"] == food_name]["description"].values[0]
        return jsonify({"food": food_name, "description": description})
    else:
        return jsonify({"error": "Food not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
