import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split

bank_df = pd.read_csv('BankNote_Authentication.csv')
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    bank_df.iloc[:, :-1], bank_df.iloc[:, -1], test_size=0.2, random_state=42)

# Train the model on the training set
y_train = [str(y) for y in y_train]
print(bank_df.columns)
model = LogisticRegression()
model.fit(X_train, y_train)
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)