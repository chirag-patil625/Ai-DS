# Linear Regression

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df=pd.read_csv("C:/Users/BHAGWAN PATIL/Desktop/Ai-Ds/Datasets/boston_house_prices.csv")
reg=LinearRegression()

# print(df)
#Dealing with missing values
print(df.isnull().sum())

#Dealing with outliers

#Finding outliers
# sns.boxplot(df['CRIM'])
# plt.show()
# sns.boxplot(df['ZN'])
# plt.show()
# sns.boxplot(df['INDUS'])
# plt.show()
# sns.boxplot(df['CHAS'])
# plt.show()
# sns.boxplot(df['NOX'])
# plt.show()
# sns.boxplot(df['RM'])
# plt.show()
# sns.boxplot(df['AGE'])
# plt.show()
# sns.boxplot(df['DIS'])
# plt.show()
# sns.boxplot(df['RAD'])
# plt.show()
# sns.boxplot(df['TAX'])
# plt.show()
# sns.boxplot(df['PTRATIO'])
# plt.show()
# sns.boxplot(df['B'])
# plt.show()
# sns.boxplot(df['LSTAT'])
# plt.show()
# sns.boxplot(df['MEDV'])
# plt.show()

# Outlier detection and handling
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

#using the function handle_outliers
df = handle_outliers(df, 'CRIM')
df = handle_outliers(df, 'ZN')
df = handle_outliers(df, 'CHAS')
df = handle_outliers(df, 'INDUS')
df = handle_outliers(df, 'NOX')
df = handle_outliers(df, 'AGE')
df = handle_outliers(df, 'RAD')
df = handle_outliers(df, 'TAX')
df = handle_outliers(df, 'RM')
df = handle_outliers(df, 'DIS')
df = handle_outliers(df, 'B')
df = handle_outliers(df, 'LSTAT')
df = handle_outliers(df, 'MEDV')
df = handle_outliers(df, 'PTRATIO')

# Visualize the boxplots for the handled features
# features_to_visualize = ['CRIM','ZN','CHAS','INDUS','NOX','AGE','RAD','TAX','RM', 'DIS', 'B', 'LSTAT', 'MEDV', 'PTRATIO']
# for feature in features_to_visualize:
#     sns.boxplot(df[feature])
#     plt.title(feature)
#     plt.show()


# Split the data into features (X) and target variable (y)
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train the model
reg.fit(X_train, y_train)

# Predict on the test set
y_pred = reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)