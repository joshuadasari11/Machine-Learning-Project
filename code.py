import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data sheets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Convert date column to official datetime format
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

# Filter data to use 2016 and 2017 to make it faster to run
train = train[train['date'] >= '2016-01-01'].copy()

# Extract calendar numbers from the date column
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['dayofweek'] = train['date'].dt.dayofweek

test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['dayofweek'] = test['date'].dt.dayofweek

# Map product text categories into basic numbers
all_families = pd.concat([train['family'], test['family']]).unique()
family_map = {name: i for i, name in enumerate(all_families)}

train['family_code'] = train['family'].map(family_map)
test['family_code'] = test['family'].map(family_map)

# Define the features we want our model to use
features = ['store_nbr', 'family_code', 'onpromotion', 'month', 'day', 'dayofweek']

X_train = train[features]
y_train = train['sales']
X_test = test[features]

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future sales
predictions = model.predict(X_test)

# Fix any negative predictions to 0
predictions = np.clip(predictions, 0, None)

# Create the final submission dataframe
submission = pd.DataFrame({
    'id': test['id'],
    'sales': predictions
})

# Save the output to a csv file
submission.to_csv('submission.csv', index=False)
print("Finished. Submission file is ready.")