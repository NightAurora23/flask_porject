from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = pickle.load(open('house_price_model.pkl', 'rb'))

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        area = float(request.form['area'])
        bedrooms = float(request.form['bedrooms'])

        # Prepare features
        features = np.array([[area, bedrooms]])

        # Make prediction
        prediction = model.predict(features)

        # Return result
        return render_template(
            'result.html',
            prediction_text=f"Estimated Price: ₹{int(prediction[0])}"
        )
    
    except Exception as e:
        return f"Error: {str(e)}"

# Run app (IMPORTANT for Railway)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
