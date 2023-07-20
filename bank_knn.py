from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
bank_df = pd.read_csv('BankNote_Authentication.csv')

X_train, X_test, y_train, y_test = train_test_split(
    bank_df.iloc[:, :-1], bank_df.iloc[:, -1], test_size=0.2, random_state=42)
y_train = [str(y) for y in y_train]
n_neighbors = 3
model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(X_train, y_train)

with open('KNN.pkl', 'wb') as f:
    pickle.dump(model, f)