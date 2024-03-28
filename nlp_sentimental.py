import pandas as pd 
df=pd.read_csv("/content/train.csv")
df.head()
x=df["Comment"]
y=df["Topic"] 
from sklearn.feature_extraction.text import TfidfVectorizer 
vector=TfidfVectorizer(stop_words="english") 
x1=vector.fit_transform(x)
from sklearn.model_selection import train_test_split 
a,b,c,d=train_test_split(x1,y,test_size=0.2,random_state=0) 


#using ml
from sklearn.svm import SVC
obj=SVC()
obj.fit(a,c) 
ycap=obj.predict(b) 

from sklearn.metrics import accuracy_score 
print(accuracy_score(d,ycap))

new=["I'm a medication technician. And that's alot"]
new_df=vector.transform(new)
print(obj.predict(new_df))


'''
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape

# Load the dataset
df = pd.read_csv("/content/train.csv")
x = df["Comment"]
y = df["Topic"]

# Map string labels to numeric values
label_mapping = {label: idx for idx, label in enumerate(y.unique())}
y_numeric = y.map(label_mapping)

# Vectorize the text data
vector = TfidfVectorizer(stop_words="english") 
x1 = vector.fit_transform(x)

# Split the data into training and testing sets
a, b, c, d = train_test_split(x1, y_numeric, test_size=0.2, random_state=0)

# Reshape the input data to be two-dimensional
a_reshaped = a.toarray()  # Convert sparse matrix to a dense array
b_reshaped = b.toarray()

# Define and compile LSTM model
model_lstm = Sequential()
model_lstm.add(Dense(64, input_shape=(a_reshaped.shape[1],), activation="relu"))
model_lstm.add(Reshape((1, model_lstm.output_shape[1])))  # Reshape to match LSTM input shape
model_lstm.add(LSTM(64, activation="relu"))
model_lstm.add(Dense(32, activation="relu"))
model_lstm.add(Dense(1, activation="sigmoid"))
model_lstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the LSTM model
model_lstm.fit(a_reshaped, c, epochs=10)

# Make predictions using the trained model
ycap_lstm = model_lstm.predict(b_reshaped)
'''