#Immporting libaries
import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Read the JSON file
df = pd.read_json("../Datasets/Whats-Cooking.json")

# Split the data into features (X) and target variable (y)
X = df.drop(columns='cuisine')
y = df['cuisine']

#Converting Input Attributes
mlb = MultiLabelBinarizer()
encoded_data = mlb.fit_transform(df['ingredients'])
encoded_df = pd.DataFrame(encoded_data, columns=mlb.classes_)

# Replace 'ingredients' column with the one-hot encoded DataFrame
X_encoded = pd.concat([X.drop(columns='ingredients'), encoded_df], axis=1)

print(X_encoded)

#Dealing with Missing Values
# print(X_encoded.isnull().sum())

#creating Instance variable
rf=RandomForestClassifier()

#trainig
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=1)

rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)

score=accuracy_score(y_test,y_pred)

print("Random Tree:",score)