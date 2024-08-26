import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def load_dataset(file_path):
    df = pd.read_csv(file_path, encoding='latin1')
    return df

def preprocess_data(df):
    
    df['text'] = df['text'].fillna('')
    df['sentiment'] = df['sentiment'].fillna('neutral')  

    
    df['sentiment'] = df['sentiment'].astype(str)

    
    df['text'] = df['text'].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))
    return df

def train_model(X_train, y_train):
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train).toarray()

    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_train_one_hot = to_categorical(y_train_encoded)

    
    model = Sequential()
    model.add(Dense(512, input_dim=X_train_vec.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y_train_one_hot.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_vec, y_train_one_hot, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

    return model, vectorizer, label_encoder

def evaluate_model(model, vectorizer, label_encoder, X_test, y_test):
    
    X_test_vec = vectorizer.transform(X_test).toarray()

    
    y_test_encoded = label_encoder.transform(y_test)
    y_test_one_hot = to_categorical(y_test_encoded)

    
    y_pred_prob = model.predict(X_test_vec)
    y_pred = np.argmax(y_pred_prob, axis=1)

    accuracy = accuracy_score(np.argmax(y_test_one_hot, axis=1), y_pred)
    report = classification_report(np.argmax(y_test_one_hot, axis=1), y_pred, target_names=label_encoder.classes_)

    return accuracy, report

def predict_sentiment(model, vectorizer, label_encoder, text):
    
    text = " ".join([word for word in text.split() if word not in stop_words])

    
    text_vec = vectorizer.transform([text]).toarray()

    
    text_pred_prob = model.predict(text_vec)
    text_pred = np.argmax(text_pred_prob, axis=1)
    sentiment = label_encoder.inverse_transform(text_pred)

    return sentiment[0]

def main(train_file_path, test_file_path):
    
    train_df = load_dataset(train_file_path)
    test_df = load_dataset(test_file_path)

    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    
    X_train = train_df['text']
    y_train = train_df['sentiment']
    X_test = test_df['text']
    y_test = test_df['sentiment']

    
    model, vectorizer, label_encoder = train_model(X_train, y_train)

    
    accuracy, report = evaluate_model(model, vectorizer, label_encoder, X_test, y_test)

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

    
    while True:
        user_input = input("Enter a sentence to analyze sentiment (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        sentiment = predict_sentiment(model, vectorizer, label_encoder, user_input)
        print(f"The sentiment of the sentence is: {sentiment}")
