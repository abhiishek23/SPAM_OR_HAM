from flask import Flask, render_template, request
import pickle

# Load the trained model and vectorizer from the pickle files
model_path = r'Model/modelforPrediction.pkl'
vectorizer_path = r'Model/vectorizer.pkl'

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize the Flask application
app = Flask(__name__)

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the message from the form
        message = request.form['message']
        
        # Transform the message using the loaded vectorizer
        data = [message]
        vect = vectorizer.transform(data).toarray()
        
        # Make the prediction
        prediction = model.predict(vect)
        
        # Determine the output label
        result = "Spam" if prediction[0] == 1 else "Ham"
        
        # Render the result on the webpage
        return render_template('result.html', prediction=result, message=message)

if __name__ == "__main__":
    app.run(debug=False)

