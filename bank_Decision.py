import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the Iris dataset
bank_df = pd.read_csv('BankNote_Authentication.csv')
bank_df.columns
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    bank_df.iloc[:, :-1], bank_df.iloc[:, -1], test_size=0.2, random_state=42)

# Train the model on the training set
y_train = [str(y) for y in y_train]
print(bank_df.columns)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# Save the trained model to a file
with open('DecisionTree.pkl', 'wb') as f:
    pickle.dump(model, f)