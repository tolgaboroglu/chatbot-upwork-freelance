from flask import Flask, request, jsonify, render_template
import pandas as pd
from transformers import pipeline
from pyngrok import ngrok

app = Flask(__name__)

# Load the CSV file
dilekce = pd.read_csv("/Users/tb/Documents/GitHub/case-files-chatbot/dilekcelerin.csv")

# Turkish QA model
model_name = "savasy/bert-base-turkish-squad"
qa_pipeline = pipeline("question-answering", model=model_name)

def answer_question(question, context):
    result = qa_pipeline(question=question, context=context, max_answer_len=100)
    return result['answer']

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        data = request.get_json()
        if 'user_input' not in data:
            return jsonify({"response": "No input provided"}), 400
        user_input = data['user_input']
        context = " ".join(dilekce['IctihatMetni'].dropna().astype(str).sample(n=5).tolist())
        print("Context:", context)  # Debug print to check the context
        response = answer_question(user_input, context)
        print("Response:", response)  # Debug print to check the response
        return jsonify({"response": response})
    except Exception as e:
        print("Error:", str(e))  # Print the error message for debugging
        return jsonify({"response": "An error occurred. Please try again later."}), 500

if __name__ == '__main__':
    # Start the ngrok tunnel
    public_url = ngrok.connect(5000)
    print(f"Flask app is accessible at this URL: {public_url}")

    # Start the Flask app
    app.run(port=5000)
