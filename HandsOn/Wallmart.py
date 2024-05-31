import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

# Generate example data for demonstration
true_values = np.array([1, 2, 3, 4, 5])
predicted_values = np.array([1.2, 1.8, 2.9, 3.7, 5.2])

# Compute mean squared error for example data
mse_example = mean_squared_error(true_values, predicted_values)
print("Mean Squared Error (example):", mse_example)

# Read the dataset
file_path ="../Datasets/Wallmart.csv"
df = pd.read_csv(file_path)

# Handling missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Plot boxplot function
def plot_boxplot(data, column):
    sns.boxplot(y=data[column])
    plt.title(f"Boxplot of {column}")
    plt.show()

columns_to_plot = ['Store', 'Dept', 'Weekly_Sales', 'IsHoliday']
for column in columns_to_plot:
    plot_boxplot(df, column)

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Extract date features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['WeekOfYear'] = df['Date'].dt.isocalendar().week

# Drop the original 'Date' column
df = df.drop('Date', axis=1)

# Prepare features and target variable
X = df.drop('Weekly_Sales', axis=1)
y = df['Weekly_Sales']

# Define the column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Store', 'Dept', 'Year', 'Month', 'Day', 'WeekOfYear']),
        ('cat', OneHotEncoder(drop='first'), ['IsHoliday'])
    ])

# Create a pipeline with preprocessing and linear regression
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on the test data
y_pred = pipeline.predict(X_test)

# Calculate Mean Squared Error on predictions
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (pipeline):", mse)

# Cross-validation for better evaluation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
cv_mse = -cv_scores.mean()
print("Cross-validated Mean Squared Error:", cv_mse)