import pandas as pd
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the model
model_path = 'models_pkl_files/GradientBoosting.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Extract feature names from the model (assuming scikit-learn)
model_features = model.feature_names_in_

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        year = int(request.form['year'])
        crop = request.form['crop']
        
        # Create a DataFrame for the input
        input_data = pd.DataFrame({'Year': [year], 'Crop': [crop]})
        input_data = pd.get_dummies(input_data, columns=['Crop'])
        
        # Ensure all possible crop columns are present and in the correct order
        for feature in model_features:
            if feature not in input_data.columns:
                input_data[feature] = 0
        input_data = input_data[model_features]  # Ensure the order matches the training data
        
        prediction = model.predict(input_data)[0]
        return render_template('index.html', prediction_text=f'Predicted Production: {prediction:.2f}')
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
