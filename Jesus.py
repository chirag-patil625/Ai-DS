import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Read the dataset
df = pd.read_csv("C:/Users/BHAGWAN PATIL/Desktop/Ai-Ds/Datasets/winequalityN.csv")

# Drop the 'type' column as it seems to be categorical and not relevant for regression
df.drop(columns=['type'], inplace=True)

def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR

    df[column] = df[column].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))
    return df

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
for column in numeric_columns:
    df = handle_outliers(df, column)
# Fill missing values with mean for numeric columns
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].mean())


# Split the data into features (X) and target variable (y)
X = df.drop(columns='quality')
y = df['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train the model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict on the test set
y_pred = reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
