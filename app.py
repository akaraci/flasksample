from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

# Hazır Iris veri setini yükle
iris = load_iris()
X = iris.data
y = iris.target

# Modeli eğit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        sample = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0][prediction] * 100
        result = iris.target_names[prediction]

        return render_template(
            "index.html",
            result=result,
            probability=f"{probability:.2f}"
        )

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            error="Lütfen bütün alanlara geçerli sayılar giriniz."
        )

if __name__ == "__main__":
    app.run(debug=True)
