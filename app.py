from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import certifi
import requests
import json
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/process": {"origins": "http://localhost:3000"}})  # Enable CORS for the /process route

# Environment variables (Normally these should be securely stored)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
API_URL = os.getenv("API_URL")

# Initialize MongoDB client
ca = certifi.where()
mongo_client = MongoClient(MONGODB_URI, tlsCAFile=ca)
db = mongo_client['Chatbot']
collection = db['Duke2']

# Initialize OpenAI client
embedding_model = SentenceTransformer("avsolatorio/GIST-large-Embedding-v0")

def embed_message(user_message):
    """
    embeds the user_message using Gist.
    
    :param user_message: The user message to embed. Type string.
    :return: The embedded message. Type list.
    """
    message_embedding = embedding_model.encode([user_message], convert_to_tensor=True).tolist()[0]
    return message_embedding

def find_similar_chunks(embedded_message, max_results=3):
    query = [
        {
            "$vectorSearch": {
                "index": "vector_index2",
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
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()  # This should be a dictionary
    except Exception as e:
        print(f"Failed to get response from Hugging Face model: {str(e)}")
        return {"error": str(e)}

def extract_response(text):
    # Find the position of the marker "### Response:"
    marker = "### Response:"
    start_index = text.find(marker)
    
    # If the marker is not found, return an empty string
    if start_index == -1:
        return ""
    
    # Add the length of the marker to start index to find the start of the response
    start_of_response = start_index + len(marker)
    
    # Extract and return the text following the marker
    return text[start_of_response:].strip()

@app.route('/process', methods=['POST'])
@cross_origin()
def process_request():
    # Extract message from the request
    message = request.json.get("message", "")
    print(message)

    # Step 1: Get embedding
    embedding = embed_message(message)

    # Step 2: Retrieve similar chunks from MongoDB
    similar_chunks = find_similar_chunks(embedding)

    # Step 3: Call HuggingFace model
    if similar_chunks:
        print(similar_chunks)
        # Assuming we use the text from the first similar chunk
        model_input = similar_chunks[0] + " " + message
        huggingface_response = query_huggingface_model({"inputs": model_input})
        print(huggingface_response)
        text_response = extract_response(huggingface_response)
        print(text_response)
        return jsonify(text_response), 200
    else:
        return jsonify({"message": "No similar chunks found."}), 404

if __name__ == '__main__':
    app.run(debug=True)
