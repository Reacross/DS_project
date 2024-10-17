from flask import Flask, render_template, request
import joblib

app = Flask(__name__)


model = joblib.load('example.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        comment = request.form['comment']
        prediction = model.predict([comment])[0]
        result = 'Toxic' if prediction == 1 else 'Not Toxic'
        return render_template('result.html', comment=comment, result=result)

if __name__ == '__main__':
    app.run(debug=True)
