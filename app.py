from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from sentence_transformers import SentenceTransformer, util
import numpy as np
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import BulkWriteError
import json

# Initialize Flask app and MongoDB client
app = Flask(__name__)
api = Api(app)

uri = "mongodb+srv://theshaikhasif03:fPQSb56RBLe2lG84@cluster1.o65jh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"
client = MongoClient(uri, server_api=ServerApi("1"))

# Send a ping to confirm a successful connection
try:
    client.admin.command("ping")
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# MongoDB setup
db = client["test"]
faq_collection = db["faqs"]  # Collection for FAQs

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


class FAQ(Resource):
    def get(self):
        """Endpoint to get all FAQs from the database."""
        faqs = list(faq_collection.find({}, {"_id": 0}))  # Exclude MongoDB's _id
        return jsonify(faqs)

    def post(self):
        """Endpoint to add a new FAQ."""
        new_faq = request.json
        faq_collection.insert_one(new_faq)
        return jsonify({"message": "FAQ added successfully!"})


class Chatbot(Resource):
    def post(self):
        """Endpoint to query chatbot based on user's question."""
        # Extract the "user_input" field from the JSON body
        user_query = request.json.get("user_input")

        # Check if the user_query is empty or missing
        if not user_query:
            return {"error": "No question provided"}, 400

        faqs = list(faq_collection.find({}, {"_id": 0}))
        if not faqs:
            return {"message": "No FAQs found in the database."}

        faq_questions = [faq["question"] for faq in faqs]
        faq_answers = [faq["answer"] for faq in faqs]

        faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)
        user_embedding = model.encode(user_query, convert_to_tensor=True)

        cosine_similarities = util.pytorch_cos_sim(user_embedding, faq_embeddings)
        best_match_idx = np.argmax(cosine_similarities)

        response = faq_answers[best_match_idx]
        return {"question": faq_questions[best_match_idx], "answer": response}


# def insert_data_from_file(file_path, collection):
#     """Function to read a data file and insert its contents into MongoDB."""
#     try:
#         with open(file_path, "r") as file:
#             data = json.load(file)

#             if isinstance(data, dict):
#                 collection.insert_one(data)
#                 return "Single document inserted successfully."
#             elif isinstance(data, list):
#                 collection.insert_many(data)
#                 return "Multiple documents inserted successfully."
#             else:
#                 return "The file content format is not supported."

#     except FileNotFoundError:
#         return "File not found."
#     except json.JSONDecodeError:
#         return "Error decoding the JSON file."
#     except BulkWriteError as e:
#         return f"Error inserting documents: {e.details}"
#     except Exception as e:
#         return f"An error occurred: {str(e)}"


# # Example usage:
# file_path = "faq.json"  # Path to your data file
# result_message = insert_data_from_file(file_path, faq_collection)
# print(result_message)

# Register resources
api.add_resource(FAQ, "/faq")
api.add_resource(Chatbot, "/chatbot")

if __name__ == "__main__":
    app.run(debug=True)
