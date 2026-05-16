from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("cp2.pkl")
model_columns = joblib.load("rf_columns.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    age = int(request.form['age'])
    bmi = float(request.form['bmi'])
    bloodpressure = int(request.form['bloodpressure'])
    dependents = float(request.form['dependents'])
    smoker = int(request.form['smoker'])
    diabetes = int(request.form['diabetes'])
    regular_ex = int(request.form['regular_ex'])
    sex = int(request.form['sex'])
    hereditary_diseases = int(request.form['hereditary_diseases'])

    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'bloodpressure': [bloodpressure],
        'dependents': [dependents],
        'smoker': [smoker],
        'diabetes': [diabetes],
        'regular_ex': [regular_ex],
        'sex': [sex],
        'hereditary_diseases': [hereditary_diseases]
    })

    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )

    prediction = model.predict(input_data)[0]

    return render_template(
        'index.html',
        prediction_text=f'Predicted Claim Amount: ${prediction:.2f}'
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)