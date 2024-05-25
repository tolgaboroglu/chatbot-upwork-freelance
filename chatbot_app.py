from flask import Flask, request, jsonify, render_template
import pandas as pd
from transformers import pipeline
from pyngrok import ngrok

app = Flask(__name__)

# CSV dosyasını yükleme
dilekce = pd.read_csv("dilekcelerin.csv")

# Türkçe soru cevaplama modeli
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
    user_input = request.form['user_input']
    context = " ".join(dilekce['IctihatMetni'].dropna().astype(str).sample(n=5).tolist())  # Rastgele 5 bağlam seçip birleştiriyoruz
    response = answer_question(user_input, context)
    return jsonify({"response": response})

if __name__ == '__main__':
    # Ngrok tünelini başlatma ve URL'yi alma
    public_url = ngrok.connect(5000)
    print(f"Flask uygulamasına erişmek için bu URL'yi kullanın: {public_url}")

    # Flask uygulamasını başlatma
    app.run(port=5000)