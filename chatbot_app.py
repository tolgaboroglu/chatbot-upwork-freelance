from flask import Flask, request, jsonify, render_template
import pandas as pd
from transformers import pipeline
from pyngrok import ngrok

app = Flask(__name__)

# CSV dosyasını yükleme
dilekce = pd.read_csv("file.csv")

# Türkçe soru cevaplama modeli
model_name = "dbmdz/bert-base-turkish-cased"
qa_pipeline = pipeline("question-answering", model=model_name)

# Rastgele 10 bağlam seçip birleştiriyoruz ve sabitliyoruz
context = " ".join(dilekce['target_text'].dropna().astype(str).sample(n=10).tolist())

def complete_sentence(text):
    # Tamamlanmamış cümleleri bul ve tamamla
    if text and text[-1] not in '.!?':
        last_sentence_end = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_sentence_end != -1:
            text = text[:last_sentence_end + 1]
    return text

def extend_answer_length(answer, context, min_length=300):
    words = answer.split()
    if len(words) >= min_length:
        return answer

    remaining_context = context.replace(answer, "")
    additional_text = remaining_context[:min_length * 5]  # Geniş bir tampon alan alıyoruz
    extended_answer = answer + additional_text

    extended_answer = complete_sentence(extended_answer)

    return extended_answer

def answer_question(question, context, min_length=300):
    result = qa_pipeline(question=question, context=context, max_answer_len=500, min_answer_len=250)
    answer = result['answer']
    extended_answer = extend_answer_length(answer, context, min_length=min_length)
    return extended_answer

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.get_json()
    if 'user_input' in data:
        user_input = data['user_input']
        response = answer_question(user_input, context)
        return jsonify({"response": response})
    else:
        return jsonify({"error": "User input not found"}), 400

if __name__ == '__main__':
    # Ngrok tünelini başlatma ve URL'yi alma
    public_url = ngrok.connect(5000)
    print(f"Flask uygulamasına erişmek için bu URL'yi kullanın: {public_url}")

    # Flask uygulamasını başlatma
    app.run(port=5000)