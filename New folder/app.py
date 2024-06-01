from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('flight_rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form submission
    source = request.form['source']
    destination = request.form['destination']
    date_of_journey = request.form['date_of_journey']
    
    # Convert the data into a format suitable for prediction
    input_data = prepare_input(source, destination, date_of_journey)
    
    # Perform prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Convert the prediction to a list
    predicted_fare = prediction.tolist()
    
    # Render the result template with the predicted fare
    return render_template('result.html', predicted_fare=predicted_fare)

def prepare_input(source, destination, date_of_journey):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({'Source': [source],
                               'Destination': [destination],
                               'Date_of_Journey': [date_of_journey]})
    
    # Preprocess the input data similar to what was done during training
    # For simplicity, you may need to preprocess date_of_journey
    
    return input_data

if __name__ == '__main__':
    app.run(debug=True)
