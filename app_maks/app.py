import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, render_template, request, jsonify

from comment_summarizer import summarizer as sm

app = Flask(__name__)

# Завантаження повної моделі BERT
model_path = "bert_toxicity_full_model.pt"
if not os.path.exists(model_path):
    print(f"Модель не знайдено за шляхом: {os.path.abspath(model_path)}")
    print("Спробуємо завантажити модель з Hugging Face...")
    model = BertForSequenceClassification.from_pretrained("unitary/toxic-bert")
    torch.save(model, model_path)
    print(f"Модель збережено за шляхом: {os.path.abspath(model_path)}")
else:
    model = torch.load(model_path, map_location=torch.device("cpu"))

model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def classify_text(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=256
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.sigmoid(outputs.logits).squeeze().tolist()
    return {label: prob for label, prob in zip(labels, probabilities)}


def is_spam(comment, summary):
    # Перевірка на пустий коментар або коментар, коротший за summary
    if not comment.strip() or len(comment) / 2 > len(summary):
        return True
    return False


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        comment = request.form["comment"]
        summary = sm.summarize(comment)

        if is_spam(comment, summary):
            return jsonify(
                {
                    "original_comment": comment,
                    "classification": {"spam": 1.0},
                    "message": "Цей коментар класифіковано як спам.",
                }
            )

        classification = classify_text(comment)
        return jsonify(
            {
                "original_comment": comment,
                "summarized_comment": summary,
                "classification": classification,
            }
        )
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
