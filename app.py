from flask import Flask, request, render_template
from flask_cors import CORS
import joblib
import datetime

app = Flask(__name__)
CORS(app)

# Load the trained model pipeline
model = joblib.load("car_price_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form

        present_price = float(form_data['Present_Price'])
        fuel_type = form_data['Fuel_Type_Petrol']
        seller_type = form_data['Seller_Type_Individual']
        transmission = form_data['Transmission_Mannual']

        year = form_data.get('Year')
        kms_driven = form_data.get('Kms_Driven')
        owner = form_data.get('Owner')

        if year:
            year = int(year)
            current_year = datetime.datetime.now().year
            age = current_year - year
        else:
            age = 0

        kms_driven = int(kms_driven) if kms_driven else 0
        owner = int(owner) if owner else 0

        # Prepare input as a dictionary (raw values, no encoding)
        input_data = {
            'Present_Price': present_price,
            'Kms_Driven': kms_driven,
            'Owner': owner,
            'Car_Age': age,
            'Fuel_Type': fuel_type,
            'Seller_Type': seller_type,
            'Transmission': transmission
        }

        # Create DataFrame from one row
        import pandas as pd
        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]
        output = round(prediction, 2)

        return render_template('index.html', prediction_text=f"Estimated Selling Price: ₹ {output} Lakhs")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
