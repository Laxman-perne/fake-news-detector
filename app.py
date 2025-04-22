import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from flask import Flask, request, render_template
import os

data = pd.read_csv('news.csv')  
X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

y_pred = model.predict(tfidf_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']
    vectorizer = joblib.load('vectorizer.pkl')
    model = joblib.load('model.pkl')
    news_vec = vectorizer.transform([news_text])
    prediction = model.predict(news_vec)
    return render_template('index.html', prediction_text=f'This news is {prediction[0]}')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)