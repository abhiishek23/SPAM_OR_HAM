# Spam-Ham Classifier using Natural Language Processing (NLP)

## Project Overview

This project is a Spam-Ham classifier built using Natural Language Processing (NLP) techniques. The classifier is trained to predict whether a given SMS message is "Spam" or "Ham" (non-spam) using a Naive Bayes model. Two different approaches were explored: Bag of Words (BOW) and TF-IDF (Term Frequency-Inverse Document Frequency). The BOW model was found to perform better in terms of accuracy.

## Dataset

- **Source**: [SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- The dataset contains a collection of 5,574 SMS messages labeled as either "ham" (legitimate) or "spam".

## Requirements

To run this project, you need the following Python libraries:

```plaintext
Flask==2.3.2
nltk==3.8.1
scikit-learn==1.2.2
pandas==1.5.3
```

## Project Structure

```
D:\\PW Projects\\Spam-Classifier-NLP\\
│
├── application.py        # Flask web application
├── Model
│   ├── modelforPrediction.pkl  # Trained Naive Bayes model (BOW)
│   └── vectorizer.pkl          # CountVectorizer (BOW)
└── templates
    ├── index.html         # HTML template for user input
    └── result.html        # HTML template for displaying prediction result
```

## Implementation Details

### 1. **Data Preprocessing**

- **Text Cleaning**: Removal of non-alphabetical characters, conversion to lowercase, and stemming of words using the `PorterStemmer`.
- **Stopwords Removal**: Common words that do not contribute much to the model’s predictions are removed using NLTK’s stopwords list.
- **Corpus Creation**: A corpus of preprocessed messages is created for further analysis.

### 2. **Model Building**

- **Bag of Words (BOW)**: 
  - A `CountVectorizer` with a maximum of 2500 features is used to convert text data into numerical form.
  - The data is then split into training and testing sets.
  - A `MultinomialNB` classifier is trained on the BOW-transformed data.
  - The BOW model achieved higher accuracy compared to TF-IDF.

- **TF-IDF**: 
  - A `TfidfVectorizer` with a maximum of 2500 features is used to convert text data into numerical form.
  - Similar training and testing processes are followed as with BOW.
  - The TF-IDF model achieved slightly lower accuracy compared to BOW.

### 3. **Evaluation**

The models are evaluated using accuracy, confusion matrix, and classification report metrics.

### 4. **Saving the Model**

- The trained BOW model and its corresponding vectorizer are saved using `pickle` for later use in the Flask web application.

### 5. **Flask Web Application**

- A simple Flask web application is created to allow users to input an SMS message and get a prediction on whether it's Spam or Ham.
- The application loads the pre-trained model and vectorizer to make predictions in real-time.

## Running the Project

### 1. **Start the Flask Application**

Run the Flask web application:

```bash
python application.py
```

Then, open your web browser and go to `http://127.0.0.1:5000/`.

## Author

- **Abhishek Singh**
- **abhisinghbest30@gmail.com**
