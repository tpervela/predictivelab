from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and preprocessing tools
model = joblib.load('model.pkl')
sc_x = joblib.load('sc_x.pkl')
sc_y = joblib.load('sc_y.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        required_fields = [
            "manufacturer", "model_name", "category", "screen_size", 
            "screen", "cpu", "ram", "_storage", "gpu", 
            "operating_system", "operating_system_version", "weight"
        ]
        
        # Validate all required fields are present
        if not all(field in request.form for field in required_fields):
            missing_fields = [field for field in required_fields if field not in request.form]
            return render_template('index.html', prediction=f"Missing fields: {', '.join(missing_fields)}")
        
        # Prepare data dictionary
        data = {field: request.form[field] for field in required_fields}

        # Convert to DataFrame
        input_data = pd.DataFrame([data])

        # Encode categorical features
        for col, le in label_encoders.items():
            if col in input_data.columns:
                input_data[col] = input_data[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0]
                )

        # Convert weight from "kg" to float (if needed)
        input_data["weight"] = input_data["weight"].str.replace("kg", "").astype(float)

        # Scale numerical features
        input_data_scaled = sc_x.transform(input_data)

        # Make prediction
        prediction_scaled = model.predict(input_data_scaled)
        prediction = sc_y.inverse_transform(prediction_scaled.reshape(-1, 1))

        # Return result to the template
        return render_template('index.html', prediction=f"Predicted Price: ${round(prediction[0, 0], 2)}")

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
