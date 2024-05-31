import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from collections import Counter

# Read the dataset
df = pd.read_csv("../Datasets/Company-Bankruptcy.csv")

# Display missing values
missing_values = df.isnull().sum()
print("Missing values in the dataset:")
print(missing_values)

# Handle missing values by imputing with the mean (for simplicity)
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Split the dataset into features and target variable
X = df_imputed.drop('Bankrupt?', axis=1)
y = df_imputed['Bankrupt?']

# Oversampling using RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)

print("Original class distribution:", Counter(y))
print("Resampled class distribution:", Counter(y_resampled))

# Split the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

# Initialize and train a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate and print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)


print("Accuracy:", accuracy)
