from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('house_price_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bedrooms = float(request.form['bedrooms'])

    features = np.array([[area, bedrooms]])
    prediction = model.predict(features)

    return render_template('result.html', 
                           prediction_text=f"Estimated Price: ₹{int(prediction[0])}")

if __name__ == "__main__":
    app.run(debug=True)