import pandas as pd
# import sweetviz as sv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#Reading Dataset
df=pd.read_csv("../Datasets/Creditcard-Fraud-Detection.csv")
print(df)
print(df.isnull().sum())

print(df['Amount'])
#Fitting the data  (Over N Under)
sc = StandardScaler()
df['Amount'] = sc.fit_transform(pd.DataFrame(df['Amount']))
print(df['Amount'])

#Spliting X and Y
df = df.drop(['Time'], axis =1)
print(df['Class'].value_counts())
X = df.drop('Class', axis = 1)
y = df['Class']

#Droping dublicate values
df = df.drop_duplicates()



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create a random forest classifier
rf = RandomForestClassifier()

# Train the model
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



# # Generate a Sweetviz report
# report = sv.analyze(df)
#
# # Display the report in a web browser
# report.show_html('sweetviz_report.html')