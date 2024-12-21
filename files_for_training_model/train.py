import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pickle

lpred_train = pd.read_csv(r'C:\Users\visha\OneDrive\Desktop\GLIM materials\Term 5\ML Lab\laptops_train.csv')
lpred_test = pd.read_csv(r'C:\Users\visha\OneDrive\Desktop\GLIM materials\Term 5\ML Lab\laptops_test.csv')

lpred_train.shape

lpred = pd.concat([lpred_train, lpred_test])

lpred.columns

lpred.columns = lpred.columns.str.lower().str.replace(' ', '_')

lpred.isna().sum()

lpred['operating_system_version'].unique()

frequency = lpred['operating_system_version'].value_counts()

# Calculate percentages
percentage = lpred['operating_system_version'].value_counts(normalize=True) * 100

# Combine into a single DataFrame
result = pd.DataFrame({'Count': frequency, 'Percentage': percentage})

# Display result
print(result)

lpred['operating_system_version']=lpred['operating_system_version'].fillna(lpred['operating_system_version'].mode()[0])

print(lpred['operating_system_version'].isna().sum())

sns.countplot(data=lpred, x="operating_system_version")

lpred['weight'].replace(to_replace='kg', value='', regex=True, inplace=True)
lpred['weight'].replace(to_replace='s', value='', regex=True, inplace=True)

lpred['weight'] = lpred['weight'].astype("float64")

print(lpred.dtypes)

lpred.duplicated().sum()

lpred[lpred.duplicated()]

# Initialize LabelEncoder
le = LabelEncoder()

# Create a dictionary to hold all the encoders
encoders = {}

# Fit and transform columns, storing each encoder
encoders["manufacturer"] = le.fit(lpred["manufacturer"])
lpred["manufacturer"] = le.transform(lpred["manufacturer"])

encoders["model_name"] = le.fit(lpred["model_name"])
lpred["model_name"] = le.transform(lpred["model_name"])

encoders["category"] = le.fit(lpred["category"])
lpred["category"] = le.transform(lpred["category"])

encoders["screen_size"] = le.fit(lpred["screen_size"])
lpred["screen_size"] = le.transform(lpred["screen_size"])

encoders["screen"] = le.fit(lpred["screen"])
lpred["screen"] = le.transform(lpred["screen"])

encoders["cpu"] = le.fit(lpred["cpu"])
lpred["cpu"] = le.transform(lpred["cpu"])

encoders["ram"] = le.fit(lpred["ram"])
lpred["ram"] = le.transform(lpred["ram"])

encoders["_storage"] = le.fit(lpred["_storage"])
lpred["_storage"] = le.transform(lpred["_storage"])

encoders["gpu"] = le.fit(lpred["gpu"])
lpred["gpu"] = le.transform(lpred["gpu"])

encoders["operating_system"] = le.fit(lpred["operating_system"])
lpred["operating_system"] = le.transform(lpred["operating_system"])

encoders["operating_system_version"] = le.fit(lpred["operating_system_version"])
lpred["operating_system_version"] = le.transform(lpred["operating_system_version"])

# Save the encoders to a pickle file
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)


print(lpred.dtypes)

x = lpred.iloc[:, :-1].values
y = lpred.iloc[:, -1].values
y = y.reshape(-1, 1)

print(x.shape)

print(y.shape)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 15, random_state = 0)
model.fit(x_train, y_train)


y_pred_rft = model.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_rft.reshape(len(y_pred_rft),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import r2_score
r2_score(y_test, y_pred_rft)


import joblib

# Save the trained Random Forest model
joblib.dump(model, 'model.pkl')

# Save the feature scalers
joblib.dump(sc_x, 'sc_x.pkl')
joblib.dump(sc_y, 'sc_y.pkl')





















