import numpy as np
from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Data untuk Linear Regression (Berat vs Tinggi)
X = np.array([[60], [65], [70], [75], [80]])
y = np.array([160, 165, 170, 175, 180])

# Melatih model Linear Regression
model = LinearRegression()
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if 'weight' not in data:
            return jsonify({'error': 'No weight provided'}), 400

        weight = np.array([[float(data['weight'])]])
        prediction = model.predict(weight)
        
        return jsonify({'predicted_height': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
