from flask import Flask,request,jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))


# Define function to preprocess input data
def preprocess_input(project_duration, target_amount, no_of_backers, backers_feedback, previous_record):
    # Create a DataFrame from user input
    data = pd.DataFrame({
        'project_duration': [project_duration],
        'target_amount': [target_amount],
        'no_of_backers': [no_of_backers],
        'backers_feedback': [backers_feedback],
        'previous_record': [previous_record]
    })

    # Perform one-hot encoding
    data_encoded = pd.get_dummies(data,columns=['project_duration', 'target_amount', 'no_of_backers', 'backers_feedback','previous_record'])

    # Reindex the DataFrame to match the training data columns
    data_encoded = data_encoded.reindex(
        columns=['project_duration_Long', 'project_duration_Medium', 'project_duration_Small',
                 'target_amount_Average', 'target_amount_High', 'target_amount_Low',
                 'no_of_backers_Less than 1000', 'no_of_backers_More than 1000',
                 'backers_feedback_Bad', 'backers_feedback_Good',
                 'previous_record_No', 'previous_record_Yes'], fill_value=0)

    return data_encoded


@app.route('/')
def hello():
    return 'Hello, World!'


@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    project_duration = request.form.get('project_duration')
    target_amount = request.form.get('target_amount')
    no_of_backers = request.form.get('no_of_backers')
    backers_feedback = request.form.get('backers_feedback')
    previous_record = request.form.get('previous_record')

    # Preprocess the input data
    input_data = preprocess_input(project_duration, target_amount, no_of_backers, backers_feedback, previous_record)

    # Make prediction using the trained model
    result = model.predict(input_data)

    # Return the prediction result
    if result[0] == 1:
        return jsonify({'result': 'Successful'})
    else:
        return jsonify({'result': 'Unsuccessful'})


if __name__ == '__main__':
    app.run(debug=True)
