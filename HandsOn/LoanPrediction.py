import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# Read the dataset
df = pd.read_csv("../Datasets/Loan-Prediction.csv")

# Dealing with missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['CoapplicantIncome'].fillna((df['CoapplicantIncome'].mean()), inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)

# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])

#Dealing with Outliers
# sns.boxplot(df['ApplicantIncome'])
# plt.show()
# sns.boxplot(df['CoapplicantIncome'])
# plt.show()
# sns.boxplot(df['LoanAmount'])
# plt.show()
# sns.boxplot(df['Loan_Amount_Term'])
# plt.show()


def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound= Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    df[column] = df[column].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))
    return df

    sns.boxplot(df[column])
    plt.show()

df = handle_outliers(df, 'ApplicantIncome')
df = handle_outliers(df, 'CoapplicantIncome')
df = handle_outliers(df, 'LoanAmount')
df = handle_outliers(df, 'Loan_Amount_Term')


# Split the data into features (X) and target variable (y)
X = df.drop(columns=['Loan_ID', 'Loan_Status'])
y = df['Loan_Status']

from imblearn.over_sampling import SMOTE
sms=SMOTE(random_state=0)                  #Systemetic Minority Oversampling
X,y=sms.fit_resample(X,y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Over and Under sampling
# # Oversampling
# smote = SMOTE(random_state=0)
# X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)

# Undersampling
rus = RandomUnderSampler(random_state=1)
X_train_undersampled, y_train_undersampled = rus.fit_resample(X_train, y_train)



# Create a random forest classifier
rf = RandomForestClassifier()

# Train the model
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
