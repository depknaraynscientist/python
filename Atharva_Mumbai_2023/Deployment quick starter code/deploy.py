# Example using Flask
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('lr_model_2.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Assuming your input features are in 'data'
    prediction = model.predict([data['inputValues']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(port=5000)
