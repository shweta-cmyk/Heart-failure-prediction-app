# Heart-failure-prediction-app


from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_input = [np.array(features)]
    prediction = model.predict(final_input)
    output = "Positive for Heart Disease" if prediction[0] == 1 else "Negative for Heart Disease"
    return render_template('index.html', prediction_text=f'Result: {output}')

if __name__ == '__main__':
    app.run(debug=True)
