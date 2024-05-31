import pandas as pd
import sweetviz as sv

#Reading Dataset
df=pd.read_csv("C:/Users/BHAGWAN PATIL/Desktop/Ai-Ds/Datasets/bank_transactions.csv")
print(df)


#Dealing with missing values
print(df.isnull().sum())

df['CustGender'] = df['CustGender'].fillna(df['CustGender'].mode()[0])
df['CustLocation'] = df['CustLocation'].fillna(df['CustLocation'].mode()[0])
df['CustAccountBalance'].fillna(df['CustAccountBalance'].mean(), inplace=True)
df['CustomerDOB'].fillna(df['CustomerDOB'].mode()[0], inplace=True)

print(df.isnull().sum())


# Assuming the smartwatch is targeted towards adults aged 16-40
# We'll also assume that people with higher account balances might be more likely to purchase the smartwatch

potential_market = df[(df['CustomerDOB'] > '1983-01-01') &
                      (df['CustomerDOB'] < '2008-01-01') &
                      (df['CustAccountBalance'] > 30000) &
                      (df['TransactionAmount (INR)'] > 15000)]

# You can further filter the potential market based on other factors like location, gender, etc.

print("more likely to purchase the smartwatch:", len(potential_market))





# # Generate a Sweetviz report
# report = sv.analyze(df)
#
# # Display the report in a web browser
# report.show_html('sweetviz_report.html')