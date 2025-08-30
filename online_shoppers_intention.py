import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('/Users/ujjwalkumar/Desktop/ML/online_shoppers_intention.csv')
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Encode categorical variables
le = LabelEncoder()
data['Month'] = le.fit_transform(data['Month'])
data['VisitorType'] = le.fit_transform(data['VisitorType'])
data['Weekend'] = le.fit_transform(data['Weekend'])
data['Revenue'] = le.fit_transform(data['Revenue'])
print(data.dtypes)



if sklearn.__version__ >= "1.2":
    Ohe = OneHotEncoder(sparse_output=False)
else:
    Ohe = OneHotEncoder(sparse=False)
# Ohe = OneHotEncoder(sparse= False)
    data['Month'] = Ohe.fit_transform(data['Month'])
    data['VisitorType'] = Ohe.fit_transform(data['VisitorType'])
    data = pd.concat([data.drop(['Month','VisitorType'], axis=1), data['month','VisitorType'],],axis=1)

# Feature selection
features = ['VisitorType', 'Month']
scaler = StandardScaler()
X = scaler.fit_transform(data[features])
y = data['Revenue']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Training set size: {X_train.shape}, Test set size: {X_test.shape}')

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
cfm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cfm, annot =True, fmt = 'd', cmap='Blues'
            , xticklabels=['No Purchase', 'Purchase'], yticklabels=['No Purchase', 'Purchase'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
print(classification_report(y_test, y_pred))

# Feature importance
importance = model.coef_[0]
feature_importance = pd.Series(importance, index=features).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

try:
    print("---------------------Enter Input Data--------------------")
    input  = {
        'VisitorType': input("enter VisitorType data"),  
        'Month': input("enter Month data")
    }
    input_data = pd.DataFrame([input])
    input_scale = scaler.fit_transform((input_data))
    predict = model.predict(input_scale)[0]
    result = "TRUE" if predict == 1 else "FALSE"
    print(f"Your predicted output: {result}")
except Exception as e:
    print("An error occured:",e)