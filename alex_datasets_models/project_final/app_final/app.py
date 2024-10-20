import torch
from transformers import BertTokenizer, AlbertForSequenceClassification, AlbertTokenizer
from flask import Flask, render_template, request, jsonify

from comment_summarizer import summarizer as sm

app = Flask(__name__)

# Завантаження повної моделі ALBERT
model_path = 'app_final/toxic_albert_model_0'
# model = torch.load("app_final/toxic_albert_model_complete.pt", map_location=torch.device("cpu"))
# model = torch.jit.load("app_final/toxic_albert_model_complete.pt", map_location=torch.device("cpu"))
model = AlbertForSequenceClassification.from_pretrained(model_path)
model.to(torch.device("cpu"))

model.eval()

# tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
tokenizer = AlbertTokenizer.from_pretrained(model_path)

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def classify_text(text):
    # Tokenize the input text
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=256
    )

    input_ids = inputs['input_ids']

    # Model inference with TorchScript
    with torch.no_grad():
        outputs = model(input_ids)

    # Роздруковуємо результат, щоб перевірити структуру
    print(outputs)

    # Extract logits if they are present in outputs
    if 'logits' in outputs:
        logits = outputs['logits']
    else:
        logits = outputs[0]  # Якщо 'logits' немає, використовуємо перший елемент.

    # Compute probabilities using sigmoid
    probabilities = torch.sigmoid(logits).squeeze().tolist()

    # Map probabilities to labels
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
                    "message": "Цей коментар пустий або класифіковано як спам.",
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
    app.run(host="0.0.0.0", port=5000, debug=True)
