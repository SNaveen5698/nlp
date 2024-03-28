import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 

# Load the dataset
df = pd.read_csv("/content/SMS_train.csv", encoding='latin-1')

# Text Preprocessing
x = df["Message_body"]
y = df["Label"]

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
x_vectorized = vectorizer.fit_transform(x)

# Train-Test Split
a, b, c, d = train_test_split(x_vectorized, y, test_size=0.2, random_state=0)

# Model Training
model = LogisticRegression()
model.fit(a, c)

# Model Evaluation
y_pred = model.predict(b)
accuracy = accuracy_score(d, y_pred)
print("Accuracy:", accuracy)

# Prediction on New Data
new_data_text = ["FREE for 1st week! No1 Nokia tone 4 ur mobile"]
new_data_vectorized = vectorizer.transform(new_data_text)
prediction = model.predict(new_data_vectorized)
print("Predicted Label:", prediction[0])
