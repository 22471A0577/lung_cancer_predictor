from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("lung_cancer_model.pkl")

# Feature order used during training
feature_order = list(model.feature_names_in_)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    risk_factors = []
    safe_factors = []
    risk_percentage = 0
    error = ""

    if request.method == 'POST':
        data = []

        for feature in feature_order:
            value = request.form.get(feature)

            # AGE validation (server-side)
            if feature == "AGE":
                try:
                    age = int(value)
                    if age < 1 or age > 100:
                        error = "❌ Age must be between 1 and 100"
                        return render_template(
                            "index.html",
                            error=error,
                            feature_order=feature_order
                        )
                except:
                    error = "❌ Invalid age value"
                    return render_template(
                        "index.html",
                        error=error,
                        feature_order=feature_order
                    )

            if value is None:
                value = 0

            data.append(int(value))

        # Convert input to DataFrame
        input_df = pd.DataFrame([data], columns=feature_order)

        # Prediction
        result = model.predict(input_df)[0]

        # Risk / Safe factors
        for i, val in enumerate(data):
            if feature_order[i] not in ['AGE', 'GENDER']:
                if val == 1:
                    risk_factors.append(feature_order[i])
                else:
                    safe_factors.append(feature_order[i])

        total_features = len(feature_order) - 2
        risk_percentage = int((len(risk_factors) / total_features) * 100)

        if result == 1:
            prediction = "⚠ Lung Cancer Detected"
        else:
            prediction = "✅ No Lung Cancer Detected"

    return render_template(
        "index.html",
        prediction=prediction,
        feature_order=feature_order,
        risk_factors=risk_factors,
        safe_factors=safe_factors,
        risk_percentage=risk_percentage,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True, port=8000)
