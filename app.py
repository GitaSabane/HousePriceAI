from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model_data = pickle.load(open("house_model.pkl", "rb"))
model = model_data["model"]
columns = model_data["columns"]
metrics = model_data["metrics"]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    feature_importance = None

    if request.method == "POST":
        try:
            input_data = []
            for col in columns:
                value = float(request.form[col])
                input_data.append(value)

            df = pd.DataFrame([input_data], columns=columns)
            prediction = model.predict(df)[0]

            feature_importance = list(model.feature_importances_)

        except:
            prediction = "Invalid Input"

    return render_template(
        "index.html",
        columns=columns,
        prediction=prediction,
        metrics=metrics,
        feature_names=list(columns),
        feature_importance=feature_importance
    )

if __name__ == "__main__":
    app.run(debug=True)