from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import certifi
import requests
import json
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import re

load_dotenv()

app = Flask(__name__)
# Set the origins to the domains provided by Vercel
origins = [
    "http://localhost:3000",
    "https://aipi-chatbot-frontend-daniels-projects-a44d4a0e.vercel.app",
    "https://aipi-chatbot-frontend-git-main-daniels-projects-a44d4a0e.vercel.app",
    "https://aipi-chatbot-frontend.vercel.app",
]

CORS(app, resources={r"/process": {"origins": origins}})

# Environment variables (Normally these should be securely stored)
MONGODB_URI = os.getenv("MONGODB_URI")
MODEL_API_URL = os.getenv("MODEL_API_URL")
EMBEDDINGS_API_URL = os.getenv("EMBEDDINGS_API_URL")
MODEL_API_KEY = os.getenv("MODEL_API_KEY")
EMBEDDINGS_API_KEY = os.getenv("EMBEDDINGS_API_KEY")

# Initialize MongoDB client
ca = certifi.where()
mongo_client = MongoClient(MONGODB_URI, tlsCAFile=ca)
db = mongo_client['Chatbot']
collection = db['Duke5']
index_name = 'Duke5_index'

def embed_message(payload):
    """
    Sends a request to the specified Hugging Face model API and returns the response.
    :param payload: The data to send in the request.
    :return: The JSON response from the API.
    """
    emb_headers = {
        "Accept": "application/json",
        "Authorization": "Bearer " + EMBEDDINGS_API_KEY,
        "Content-Type": "application/json"
    }
    response = requests.post(EMBEDDINGS_API_URL, headers=emb_headers, json=payload)
    return response
    
def find_similar_chunks(embedded_message, max_results=3):
    """
    Find similar chunks in the MongoDB collection based on the embedded message.
    :param embedded_message: The embedded message to use for the search.
    :param max_results: The maximum number of results to return.
    :return: A list of similar chunks.
    """
    query = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": embedded_message,
                "numCandidates": 20,
                "limit": max_results
            }
        }
    ]
    results = list(collection.aggregate(query))
    chunks = []
    for result in results:
        chunks.append(result['chunk'])
    return chunks

def query_huggingface_model(payload):
    """
    Send request to Hugging Face model API and return the response.
    :param payload: The data to send in the request.
    :return: The JSON response from the API.
    """
    model_headers = {
        "Accept": "application/json",
        "Authorization": "Bearer " + MODEL_API_KEY,
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(MODEL_API_URL, headers=model_headers, json=payload)
        return response.json()  # This should be a dictionary
    except Exception as e:
        print(f"Failed to get response from Hugging Face model: {str(e)}")
        return {"error": str(e)}

def postprocess_response(text):
    """
    Prune all characters after the very last period in the text, including the number before the period if it exists,
    ensuring proper handling of sentences ending with class numbers or similar patterns without incorrectly identifying
    them as trailing numbers meant for pruning.
    
    :param text: The text returned by the Hugging Face model.
    :return: The response extracted from the text, pruned after the last period including the number before the period if it exists.
    """
    # Regex to find the last number-period-number sequence at the end of the text
    match = re.search(r'\d+\.\s*\d+$', text)
    if match:
        # Prune the text after this match
        text = text[:match.start()]
    else:
        # Find the last period that is not part of a sequence like "ECE 564."
        last_period_index = text.rfind('.')
        if last_period_index != -1:
            # Ensure the period is followed by space or is at the end to confirm it's a sentence end
            if last_period_index == len(text) - 1 or text[last_period_index + 1].isspace():
                text = text[:last_period_index + 1]
            else:
                # If it's not a clear sentence end, trim to the last clear sentence end
                preceding_text = text[:last_period_index].rstrip()
                last_clear_period_index = preceding_text.rfind('.')
                if last_clear_period_index != -1:
                    text = preceding_text[:last_clear_period_index + 1]

    return text.strip()

@app.route('/process', methods=['POST'])
@cross_origin()
def process_request():
    # Extract message from the request
    message = request.json.get("message", "")

    emb_payload = {
        "inputs": message,
        "parameters": {}
    }

    # Step 1: Get embedding
    response = embed_message(emb_payload)
    if response.status_code != 200:
        return jsonify({"Booting up. Please try again in a few seconds."}), 200

    # Step 2: Retrieve similar chunks from MongoDB
    similar_chunks = find_similar_chunks(response.json()['embeddings'])

    # Step 3: Call HuggingFace model
    if similar_chunks:
        # Assuming we use the text from the first similar chunk
        model_payload = {
            "inputs": "",
            "question": message,
            "context": similar_chunks[0],
            "temp": 0.3,
            "max_tokens": 180
            }
        huggingface_response = query_huggingface_model(model_payload)
        print('huggingface response:', huggingface_response)
        if 'error' in huggingface_response:
            text_response = "Booting up. Please try again in a few seconds."
        else:
            text_response = postprocess_response(huggingface_response)
        return jsonify(text_response), 200
    else:
        return jsonify({"Sorry, I'm unable to answer that question at the moment."}), 200

if __name__ == '__main__':
    app.run()
