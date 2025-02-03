#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np

# Load the datasets
train = pd.read_csv(r"C:/Users/inajgow1/Downloads/train_v9rqX0R.csv")
test = pd.read_csv(r"C:/Users/inajgow1/Downloads/test_AbJTz2l.csv")

# Display basic info about the datasets
print("Train dataset head and info:")
print(train.head())
print(train.info())

print("\nTest dataset head and info:")
print(test.head())
print(test.info())


# In[35]:


# Filling missing values for Item_Weight with mean
train['Item_Weight'].fillna(train['Item_Weight'].mean(), inplace=True)
test['Item_Weight'].fillna(test['Item_Weight'].mean(), inplace=True)

# Filling missing values for Outlet_Size with mode
train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0], inplace=True)
test['Outlet_Size'].fillna(test['Outlet_Size'].mode()[0], inplace=True)

# Confirm that there are no missing values
print("Missing values in training data:")
print(train.isnull().sum())
print("\nMissing values in testing data:")
print(test.isnull().sum())


# In[36]:


# Creating a new feature 'Years_Operational'
train['Years_Operational'] = 2023 - train['Outlet_Establishment_Year']
test['Years_Operational'] = 2023 - test['Outlet_Establishment_Year']

# Calculating mean visibility by Item_Identifier
visibility_avg = train.pivot_table(values='Item_Visibility', index='Item_Identifier')

# Creating 'Item_Visibility_MeanRatio'
train['Item_Visibility_MeanRatio'] = train.apply(
    lambda x: x['Item_Visibility'] / visibility_avg.loc[x['Item_Identifier']],
    axis=1
)
test['Item_Visibility_MeanRatio'] = test.apply(
    lambda x: x['Item_Visibility'] / visibility_avg.loc[x['Item_Identifier']],
    axis=1
)

# Log transformation for Item_MRP
train['Item_MRP_log'] = np.log1p(train['Item_MRP'])
test['Item_MRP_log'] = np.log1p(test['Item_MRP'])

# Displaying the modified datasets
print("Modified Train dataset head:")
print(train[['Outlet_Establishment_Year', 'Years_Operational', 'Item_Visibility', 'Item_Visibility_MeanRatio', 'Item_MRP', 'Item_MRP_log']].head())
print("\nModified Test dataset head:")
print(test[['Outlet_Establishment_Year', 'Years_Operational', 'Item_Visibility', 'Item_Visibility_MeanRatio', 'Item_MRP', 'Item_MRP_log']].head())


# In[37]:


from sklearn.preprocessing import OneHotEncoder

# Select categorical columns to encode
categorical_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

# Apply OneHotEncoding
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_train = encoder.fit_transform(train[categorical_cols])
encoded_test = encoder.transform(test[categorical_cols])

# Create DataFrames from encoded arrays
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_cols))
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols))

# Drop original categorical columns and concatenate the new encoded columns
train = pd.concat([train.drop(categorical_cols, axis=1), encoded_train_df], axis=1)
test = pd.concat([test.drop(categorical_cols, axis=1), encoded_test_df], axis=1)

# Display the modified datasets
print("Modified Train dataset with encoded features:")
print(train.head())
print("\nModified Test dataset with encoded features:")
print(test.head())


# In[38]:


from sklearn.model_selection import train_test_split

# Define the features (X) and the target variable (y)
X = train.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'], axis=1)
y = train['Item_Outlet_Sales']

# Split the dataset into training and validation sets (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting datasets
print(f"Training Features Shape: {X_train.shape}")
print(f"Validation Features Shape: {X_val.shape}")
print(f"Training Target Shape: {y_train.shape}")
print(f"Validation Target Shape: {y_val.shape}")


# In[39]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Initialize the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the model
gbr.fit(X_train, y_train)

# Predict on the validation set
y_pred = gbr.predict(X_val)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse}")


# In[40]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.15]
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=1
)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best model and parameters
best_gbr = grid_search.best_estimator_
best_params = grid_search.best_params_

# Predict on the validation set using the best model
y_pred_best = best_gbr.predict(X_val)

# Calculate RMSE for the best model
best_rmse = np.sqrt(mean_squared_error(y_val, y_pred_best))
print(f"Best Model Validation RMSE: {best_rmse}")
print(f"Best Model Parameters: {best_params}")


# In[41]:


import xgboost as xgb

# Initialize XGBoost Regressor with optimized parameters
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    colsample_bytree=0.5,  # Helps reduce overfitting
    learning_rate=0.05,
    max_depth=3,
    n_estimators=100,
    random_state=42
)

# Train the XGBoost model
xgb_model.fit(X_train, y_train)

# Predict on the validation set
xgb_val_pred = xgb_model.predict(X_val)

# Calculate RMSE for the XGBoost model
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
print(f"XGBoost Validation RMSE: {xgb_rmse}")


# In[42]:


# Prepare the test data (exclude identifier columns and keep features consistent)
X_test = test.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)

# Generate predictions using the trained XGBoost model
final_test_predictions = xgb_model.predict(X_test)

# Create the submission DataFrame
submission = pd.DataFrame({
    'Item_Identifier': test['Item_Identifier'],
    'Outlet_Identifier': test['Outlet_Identifier'],
    'Item_Outlet_Sales': final_test_predictions
})

# Save the submission file
submission_file_path = 'C:/Users/inajgow1/Downloads/final_submission_xgb_D5.csv'
submission.to_csv(submission_file_path, index=False)
print(f"Submission file is ready and saved as {submission_file_path}.")


# In[43]:


# Ensure no negative predictions
final_test_predictions = np.maximum(final_test_predictions, 0)

# Create the submission DataFrame
submission = pd.DataFrame({
    'Item_Identifier': test['Item_Identifier'],
    'Outlet_Identifier': test['Outlet_Identifier'],
    'Item_Outlet_Sales': final_test_predictions
})

# Save the submission file
submission_file_path = 'C:/Users/inajgow1/Downloads/final_submission_D55.csv'
submission.to_csv(submission_file_path, index=False)
print(f"Submission file with fixed predictions is ready and saved as {submission_file_path}.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


import pandas as pd

# Paths to your datasets
train_path = "C:/Users/inajgow1/Downloads/train_v9rqX0R.csv"
test_path = "C:/Users/inajgow1/Downloads/test_AbJTz2l.csv"
sample_submission_path = "C:/Users/inajgow1/Downloads/sample_submission_8RXa3c6.csv"

# Loading the data
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_submission_path)

# Displaying the first few rows of the datasets
print("Training Data - First 5 Rows:")
print(train_data.head())
print("\nTest Data - First 5 Rows:")
print(test_data.head())
print("\nSample Submission - First 5 Rows:")
print(sample_submission.head())

# Displaying summary statistics of the training data
print("\nTraining Data - Summary Statistics:")
print(train_data.describe())

# Checking for missing values in the training and test data
print("\nMissing Values - Training Data:")
print(train_data.isnull().sum())
print("\nMissing Values - Test Data:")
print(test_data.isnull().sum())


# In[ ]:





# In[2]:


# Import necessary libraries
import numpy as np

# Normalize Item_Fat_Content
train_data['Item_Fat_Content'] = train_data['Item_Fat_Content'].replace({
    'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'
})
test_data['Item_Fat_Content'] = test_data['Item_Fat_Content'].replace({
    'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'
})

# Impute missing values in Item_Weight by the mean weight of each Item_Type
item_avg_weight = train_data.pivot_table(values='Item_Weight', index='Item_Type')
def impute_weight(cols):
    Weight = cols[0]
    Type = cols[1]
    if pd.isnull(Weight):
        return item_avg_weight.loc[Type]['Item_Weight']
    return Weight

train_data['Item_Weight'] = train_data[['Item_Weight', 'Item_Type']].apply(impute_weight, axis=1)
test_data['Item_Weight'] = test_data[['Item_Weight', 'Item_Type']].apply(impute_weight, axis=1)

# Impute missing Outlet_Size with the mode of the Outlet_Type
mode_outlet_size = train_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=lambda x: x.mode()[0])
def impute_size(cols):
    Size = cols[0]
    Type = cols[1]
    if pd.isnull(Size):
        return mode_outlet_size[Type][0]
    return Size

train_data['Outlet_Size'] = train_data[['Outlet_Size', 'Outlet_Type']].apply(impute_size, axis=1)
test_data['Outlet_Size'] = test_data[['Outlet_Size', 'Outlet_Type']].apply(impute_size, axis=1)

# Check missing values again
print("Missing Values - Training Data After Imputation:")
print(train_data.isnull().sum())
print("\nMissing Values - Test Data After Imputation:")
print(test_data.isnull().sum())


# In[5]:


import datetime

# Create 'Years_Operational' feature
current_year = datetime.datetime.now().year
train_data['Years_Operational'] = current_year - train_data['Outlet_Establishment_Year']
test_data['Years_Operational'] = current_year - test_data['Outlet_Establishment_Year']

# Interaction feature between 'Item_Visibility' and 'Item_MRP'
train_data['Visibility_MRP'] = train_data['Item_Visibility'] * train_data['Item_MRP']
test_data['Visibility_MRP'] = test_data['Item_Visibility'] * test_data['Item_MRP']

# Quick look at the updated DataFrame
print(train_data.head())

# Proceed with creating a simple linear regression model as a baseline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Prepare data for training
X = train_data.drop(['Item_Outlet_Sales', 'Item_Identifier', 'Outlet_Identifier'], axis=1)
y = train_data['Item_Outlet_Sales']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f'Baseline Model RMSE: {rmse}')

# Optionally, continue to experiment with more complex models


# In[6]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Defining the categorical columns
categorical_columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 
                       'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

# Creating a ColumnTransformer to apply encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)],
    remainder='passthrough')

# Creating a pipeline that first transforms the data then fits the model
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', LinearRegression())])

# Preparing the data
X = train_data.drop(['Item_Outlet_Sales', 'Item_Identifier'], axis=1)
y = train_data['Item_Outlet_Sales']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting the model
model_pipeline.fit(X_train, y_train)

# Predicting on validation set
y_pred = model_pipeline.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f'Baseline Model RMSE with Encoding: {rmse}')


# In[7]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Creating a pipeline with preprocessing and Random Forest regressor
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# Cross-validation to evaluate the model
cv_scores = cross_val_score(rf_pipeline, X, y, cv=5, scoring='neg_root_mean_squared_error')

# Calculating average RMSE from cross-validation
average_rmse = -cv_scores.mean()
print(f'Average RMSE from CV: {average_rmse}')


# In[8]:


from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Setting up the XGBoost model pipeline
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# Parameters grid
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [3, 5, 7],
    'regressor__learning_rate': [0.01, 0.1, 0.2]
}

# Grid search with cross-validation
grid_search = GridSearchCV(xgb_pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error')
grid_search.fit(X, y)

print(f'Best RMSE: {-grid_search.best_score_}')
print('Best parameters:', grid_search.best_params_)


# In[9]:


param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [3, 5],
    'regressor__learning_rate': [0.05, 0.1],
    'regressor__min_child_weight': [1, 3],
    'regressor__subsample': [0.7, 0.9],
    'regressor__colsample_bytree': [0.7, 0.9]
}

# Re-run grid search with the new parameter grid
grid_search = GridSearchCV(xgb_pipeline, param_grid, cv=5, scoring='neg_root_mean_squared_error', verbose=1)
grid_search.fit(X, y)

print(f'Enhanced RMSE: {-grid_search.best_score_}')
print('Enhanced best parameters:', grid_search.best_params_)


# In[10]:


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

# Define base models
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
]

# Stacker model
stacker = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=100)
)

# Creating a pipeline with preprocessing and the stacker
stacking_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('stacker', stacker)
])

# Fit the model
stacking_pipeline.fit(X_train, y_train)
y_pred = stacking_pipeline.predict(X_val)
stacked_rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f'Stacked Model RMSE: {stacked_rmse}')


# In[13]:


print(f'Stacked Model RMSE: {stacked_rmse}')


# In[17]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Define categorical and numerical features
categorical_features = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
numerical_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year', 'Years_Operational', 'Visibility_MRP']

# Create transformers for categorical and numerical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())  # Optional: Use StandardScaler if model benefits from scaled data
])

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the complete pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split the dataset to avoid leakage
from sklearn.model_selection import train_test_split

# Ensure 'y' is the target variable and 'X' contains only the features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline
model_pipeline.fit(X_train, y_train)

# Predict using the fitted pipeline
predictions = model_pipeline.predict(X_test)

# Print the predicted values to confirm the setup works
print(predictions)


# In[18]:


from sklearn.metrics import mean_squared_error
import numpy as np

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse}")

# Additional metrics can be added here


# In[19]:


# Create a submission DataFrame
submission = pd.DataFrame({
    'Item_Identifier': test_data['Item_Identifier'],  # Make sure this is from the actual test set
    'Outlet_Identifier': test_data['Outlet_Identifier'],  # Same as above
    'Item_Outlet_Sales': predictions
})

# Save the submission file
submission.to_csv('final_submission_D1.csv', index=False)
print("Submission file is ready and saved as 'final_submission.csv'.")


# In[20]:


# Check the number of entries in the test data and predictions
print("Length of test data (expected for submission):", len(test_data))
print("Length of predictions made:", len(predictions))

# If they don't match, you'll need to make sure predictions are made on the full test set
if len(test_data) != len(predictions):
    print("Warning: The number of predictions does not match the number of test entries!")


# In[21]:


# Assuming 'X_full_test' is the full feature set from your test data prepared for the model
final_predictions = model_pipeline.predict(X_full_test)

# Check again
print("Length of final predictions:", len(final_predictions))


# In[22]:


# Assuming the lengths now match
if len(test_data) == len(final_predictions):
    submission = pd.DataFrame({
        'Item_Identifier': test_data['Item_Identifier'],
        'Outlet_Identifier': test_data['Outlet_Identifier'],
        'Item_Outlet_Sales': final_predictions
    })

    # Save the submission file
    submission.to_csv('final_submission.csv', index=False)
    print("Submission file is ready and saved as 'final_submission.csv'.")
else:
    print("Error: Mismatch in lengths between test data and predictions.")


# In[23]:


# Make sure your full test set is properly prepared as it should be for the model
# Assuming 'test_data' is your full test DataFrame
# Prepare 'X_full_test' by selecting the correct columns and applying any required preprocessing as done for training
X_full_test = test_data.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)  # Adjust columns as needed

# Check again the preparation
print("Columns used for predictions:", X_full_test.columns)

# Use the full test set for predictions
final_predictions = model_pipeline.predict(X_full_test)

# Confirm the length of predictions now matches the expected test entries
print("Length of final predictions:", len(final_predictions))


# In[26]:


# If lengths match, create the submission DataFrame
if len(test_data) == len(final_predictions):
    submission = pd.DataFrame({
        'Item_Identifier': test_data['Item_Identifier'],
        'Outlet_Identifier': test_data['Outlet_Identifier'],
        'Item_Outlet_Sales': final_predictions
    })

    # Save the submission file
    submission.to_csv('C:/Users/inajgow1/Downloads/final_submission_D2.csv', index=False)
    print("Submission file is ready and saved as 'final_submission.csv'.")
else:
    print("Error: There is still a mismatch in lengths between test data and predictions.")


# In[ ]:




