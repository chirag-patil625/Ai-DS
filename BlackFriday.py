import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Read the dataset
df = pd.read_csv("C:/Users/BHAGWAN PATIL/Desktop/Ai-Ds/Datasets/trainBF.csv")

# Initialize LabelEncoder
label_encoder = LabelEncoder()

print(df.isnull().sum())

# Dealing with Missing Values
df['Product_Category_1'] = df['Product_Category_1'].fillna(0)
df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
df['Product_Category_3'] = df['Product_Category_3'].fillna(0)

# Fit and transform 'Product_ID' column
df['Product_ID'] = label_encoder.fit_transform(df['Product_ID'])

# One-hot encoding for 'City_Category' column
df = pd.get_dummies(df, columns=['City_Category'], drop_first=True)

# One-hot encoding for 'Gender' column
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Define a mapping for each category to its numerical representation
age_mapping = {
    '0-17': 0,
    '18-25': 1,
    '26-35': 2,
    '36-45': 3,
    '46-50': 4,
    '51-55': 5,
    '55+': 6
}

# Map the 'Age' column using the defined mapping
df['Age'] = df['Age'].map(age_mapping)

# Define a mapping for each category to its numerical representation
stay_mapping = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4+': 4
}

# Map the 'Stay_In_Current_City_Years' column using the defined mapping
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].map(stay_mapping)

# Feature Scaling for numerical columns
scaler = StandardScaler()
numerical_columns = ['Occupation', 'Marital_Status', 'Product_Category_1', 'Product_Category_2',
                     'Product_Category_3', 'Purchase']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Define features and target variable
X = df.drop(['Purchase', 'User_ID'], axis=1)
y = df['Purchase']

# Initialize Linear Regression model
reg = LinearRegression()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

# Train the model
reg.fit(X_train, y_train)

# Predict on the test set
y_pred = reg.predict(X_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
