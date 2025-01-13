# %% [markdown]
# ## THIS PROJECT IS BASED ON DETECTION OF CREDIT CARD FRAUD USING MACHINE LEARNING ALGORITHM CALLED 'RANDOM FOREST'.
# SO IT IS A CLASSIFINATION PROBLEM.

# %%
#Let's call our data and convert it into the pandas dataframe.
import pandas as pd
raw_df = pd.read_csv('fraudTrain.csv')

# %%
#Let's take a look at the first few rows of the data.
raw_df

# %%
#Let's take a look at the shape of the data.
raw_df.shape

# %% [markdown]
# So we have total 23 features/columns and 12,96,675 instances/rows.

# %%
#Let's see some more information of this data.
pd.set_option('display.max_columns', None) #For viewing all the columns

# %%
raw_df.head()

# %%
raw_df.info()

# %%
raw_df['first'].nunique()

# %%
raw_df['last'].nunique()

# %%
raw_df['street'].nunique() 

# %%
raw_df['city'].nunique()

# %%
raw_df['zip'].value_counts()


# %%
raw_df['state'].nunique()

# %%
raw_df['job'].nunique()

# %%
raw_df['city_pop'].value_counts()

# %% [markdown]
# There are total 23 columns among them not all are important features for predicting the fraud.
# So we divide the important and non-important featues and remove the non-important featues for better prediction.

# %%
raw_df.info()

# %%
imp_feat = list(raw_df[['category', 'amt', 'gender', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long', 'is_fraud']])
non_imp_fea = list(raw_df[['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last', 'street', 'city', 'state', 'job', 'dob', 'trans_num']])

# %%
imp_feat

# %%
#Dropping the non-important columns.
raw_df.drop(columns=non_imp_fea, inplace=True)

# %%
#Modified data.
mod_raw_df = raw_df

# %%
mod_raw_df

# %%
#Information of modified data.
mod_raw_df.info()

# %%
#Statistics of modified data.
mod_raw_df.describe()

# %%
mod_raw_df['category'].nunique()

# %%
mod_raw_df['is_fraud'].value_counts()

# %% [markdown]
# So far we selected the most important featues in detecting the fraud. It's one type of feature engineering

# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %%
#Importing the libraries for visualization.
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px

#Some initials for better visualisation  
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14 
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

# %%
#As unix time is important feature for understanding the unusual transactions.
#Firtly we have to convert that into the date_time format.
mod_raw_df['date_time'] = pd.to_datetime(mod_raw_df['unix_time'], unit='s')


# %%
mod_raw_df

# %%
px.histogram(mod_raw_df,
             x = 'amt',
             title = 'AMOUNT AND FRAUD',
             color = 'is_fraud')

# %%
#Fraund between gender.
px.histogram(mod_raw_df, 
             x = 'gender',
             title = 'FRAUD AMONG FEMALE AND MALE',
             color = 'is_fraud')

# %%
#Fraud in the city
px.histogram(mod_raw_df, 
             x = 'city_pop',
             title = 'FRAUD IN THE CITY POPULATION',
             color = 'is_fraud')

# %% [markdown]
# From the upper plot it is shown that most of frauds happens in the population between 0-5000. So smaller cities have more risk of fraud.

# %%
#Fraud in a particular area using the zip code of that area
px.histogram(mod_raw_df, 
             x = 'zip',
             title = 'FRAUD IN THE ZIP AREA',
             color = 'is_fraud',
             barmode='group')

# %%
px.scatter(mod_raw_df, 
           x = 'lat',
           y = 'long',
           title = 'Longitude-Latitude',
           color = 'is_fraud')

# %%
px.histogram(mod_raw_df,
             x = 'category',
             title = 'FRAUND IN DIFFERENT CATEGORIES',
             color = 'is_fraud')

# %%
#Finally we can find the correlation between the features and can plot it!
corr_mat = mod_raw_df.select_dtypes(include = 'number').corr()

# %%
corr_mat

# %%
sns.heatmap(corr_mat, annot=True, cmap='coolwarm')

# %% [markdown]
# ## More feature engineering on UNIXTIME as it is important feature for determing the fraud

# %%
mod_raw_df['hour'] = mod_raw_df['date_time'].dt.hour
mod_raw_df['day_of_week'] = mod_raw_df['date_time'].dt.dayofweek
mod_raw_df['is_weekend'] = mod_raw_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
mod_raw_df['month'] = mod_raw_df['date_time'].dt.month
mod_raw_df['day_of_month'] = mod_raw_df['date_time'].dt.day

# %% [markdown]
# ## DATA PRE-PROCESSING

# %% [markdown]
# STEP-1: Splitting the data into training and validation set as test set is given separately
# 

# %%
mod_raw_df

# %%
#We split the train and val sets into 75:25 ratio
from sklearn.model_selection import train_test_split

# %%
train_df, val_df = train_test_split(mod_raw_df, test_size=0.25, random_state=29)

# %%
print("Training Set: ", train_df.shape)
print("Validation Set: ", val_df.shape)

# %% [markdown]
# STEP-2 Indentifying the INPUT and TARGET features

# %%
input_cols = list(mod_raw_df[['category', 'amt', 'gender', 'zip', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'hour', 'day_of_week', 'is_weekend', 'month', 
                              'day_of_month']])
target_col = list(mod_raw_df[['is_fraud']])

# %%
mod_raw_df[input_cols]

# %%
mod_raw_df[target_col]

# %% [markdown]
# We can now create inputs and targets for the training and validation for further processing and model training.

# %%
#Creating the inputs and targets for training and validation set.
#For training set:
train_inputs = train_df[input_cols].copy()
train_target = train_df[target_col].copy()

#For validation set:
val_inputs = val_df[input_cols].copy()
val_target = val_df[target_col].copy()

# %%
train_inputs

# %% [markdown]
# Identifying the numerical and categorical columns

# %%
import numpy as np 
numerical_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

# %%
train_df[numerical_cols]

# %%
train_df[categorical_cols].nunique()

# %% [markdown]
# STEP-3 Imputation on missing numerical data

# %%
#Chech if there are some missing values in the data.
train_df[numerical_cols].isna().sum()

# %% [markdown]
# So we have no missing values in the dataset, we will skip this step of IMPUTATION for now

# %% [markdown]
# STEP-4: SCALING (Modifying the values from all columns from 0 to 1 so that it helps is optimization of our model and gives better results.)

# %%
from sklearn.preprocessing import MinMaxScaler

# %%
mod_raw_df

# %%
scaler = MinMaxScaler()
scaler.fit(mod_raw_df[numerical_cols])

# %%
#Let's fit the scaler into our training and validation set
train_inputs[numerical_cols] = scaler.transform(train_inputs[numerical_cols])
val_inputs[numerical_cols] = scaler.transform(val_inputs[numerical_cols])

# %%
train_inputs[numerical_cols]

# %% [markdown]
# STEP-5: ENCODING CATEGORICAL DATA

# %%
mod_raw_df[categorical_cols].nunique()

# %%
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# %%
#Fitting our encoder into the raw data
encoder.fit(mod_raw_df[categorical_cols])

# %%
encoder.categories_

# %%
#Now we list all this into the encoded columns.   
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
print(encoded_cols)

# %%
#add new columns in train and val set
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])

# %%
train_inputs[encoded_cols]

# %%
train_inputs

# %% [markdown]
# #Pre-processing is done, now our data is ready for inserting into ML model.

# %%
#Let's check our data shape from all our datasets.  
print('train_inputs:', train_inputs.shape)
print('train_target:', train_target.shape)
print('val_inputs:', val_inputs.shape)
print('val_target:', val_target.shape)

# %% [markdown]
# ## ML Model Training and Evaluation

# %%
#Making inputs for train and val sets
X_train = train_inputs[numerical_cols + encoded_cols]
X_val = val_inputs[numerical_cols + encoded_cols]

# %%
print(X_train.shape)
print(train_target.shape)

# %%
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=-1, random_state=29)

# %%
model.fit(X_train, train_target)

# %%
#Let's predict the value for training set.
train_preds = model.predict(X_train)

# %%
train_preds

# %%
pd.value_counts(train_preds)

# %%
#Prediction probablity..
train_prob = model.predict_proba(X_train)

# %%
train_prob

# %%
#Let's see the model accuracy or score on training set..
model.score(X_train, train_target)

# %%
from sklearn.metrics import confusion_matrix
train_cof_mat = confusion_matrix(train_target, train_preds, normalize='true')
sns.heatmap(train_cof_mat, annot=True)
plt.xlabel("Predictions")
plt.ylabel("Target")
plt.title("Confusion matrix for Training data")
plt.show()

# %%
#Let's calculate F1-Score, Precision Score and Recall Score for more clarification of the model
from sklearn.metrics import f1_score, precision_score, recall_score
train_f1 = f1_score(train_target, train_preds)
train_precision = precision_score(train_target, train_preds)
train_recall = recall_score(train_target, train_preds)

# %%
print('Training F1 Score: ', train_f1)
print("Training Precision Score: ", train_precision)
print("Training Recall Score: ", train_recall)

# %% [markdown]
# It's quite nice to see the accuracy of Random Forest Algorithm on training set i.e., 99.99% 

# %% [markdown]
# Validation set calculation and analysis

# %%
val_preds = model.predict(X_val)

# %%
val_preds

# %%
pd.value_counts(val_preds)

# %%
#Let check the accuracy/score on validation set for more evulation of our model..
model.score(X_val, val_target)

# %%
val_cof_mat = confusion_matrix(val_target, val_preds, normalize='true')
sns.heatmap(val_cof_mat, annot=True)
plt.xlabel("Predictions")
plt.ylabel("Target")
plt.title("Confusion matrix for Validation data")
plt.show()

# %%
print("Model Accuracy: ", model.score(X_val, val_target)*100, "%")

# %%
#Let's calculate F1-Score, Precision Score and Recall Score for more clarification of the model
from sklearn.metrics import f1_score, precision_score, recall_score
val_f1 = f1_score(val_target, val_preds)
val_precision = precision_score(val_target, val_preds)
val_recall = recall_score(val_target, val_preds)

print("Validation F1 Score: ", val_f1)
print("Validation Precision Score: ", val_precision)
print("validation Recall Score: ", val_recall)

# %% [markdown]
# So the accuracy of our model is good and nice on Validation set also i.e., 99.81 %

# %%
#Let's check how many decision trees are in our random forest model
len(model.estimators_)

# %% [markdown]
# Just like decision tree, random forests also assign an "importance" to each feature, by combining the importance values from individual trees.

# %%
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# %%
importance_df

# %%
#Let's check top 10 importance features..
importance_df.head(10)

# %%
#Visualization of important features...
plt.title('Feature Importance')
sns.barplot(data=importance_df.head(10), x='importance', y='feature')

# %% [markdown]
# ## Model testing on test set

# %%
#Test Set calculation, analysis and predictions

#Getting the data.
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt  

#reading the csv file
test_df = pd.read_csv('fraudTest.csv')

#Important and Non-important features
imp_feat = list(test_df[['category', 'amt', 'gender', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long', 'is_fraud']])
non_imp_fea = list(test_df[['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last', 'street', 'city', 'state', 'job', 'dob', 'trans_num']])

test_df.drop(columns=non_imp_fea, inplace=True)

#Datetime conversion
test_df['date_time'] = pd.to_datetime(test_df['unix_time'], unit='s')

#More precise conversion of date_time
test_df['hour'] = test_df['date_time'].dt.hour
test_df['day_of_week'] = test_df['date_time'].dt.dayofweek
test_df['is_weekend'] = test_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
test_df['month'] = test_df['date_time'].dt.month
test_df['day_of_month'] = test_df['date_time'].dt.day

#Input and target columns
input_cols = list(test_df[['category', 'amt', 'gender', 'zip', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'hour', 'day_of_week', 'is_weekend', 'month', 
                              'day_of_month']])
target_col = list(test_df[['is_fraud']])


#test inputs and targets
test_inputs = test_df[input_cols].copy()
test_target = test_df[target_col].copy()

#Numwerical and categorical coulmns 
numerical_cols = test_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = test_inputs.select_dtypes('object').columns.tolist()

#Pre-processing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(test_df[numerical_cols])
test_inputs[numerical_cols] = scaler.transform(test_inputs[numerical_cols])

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(test_df[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

print('Test inputs:', test_inputs.shape)
print('Test target:', test_target.shape)

X_test = test_inputs[numerical_cols + encoded_cols]
print(X_test.shape)
print(test_target.shape)

#model testing-- RandomForest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=-1, random_state=29)
model.fit(X_test, test_target)
test_preds = model.predict(X_test)
test_prob = model.predict_proba(X_test)
test_acc = model.score(X_test, test_target)

#confusion matrix for test set
from sklearn.metrics import confusion_matrix
test_cof_mat = confusion_matrix(test_target, test_preds, normalize='true')
sns.heatmap(test_cof_mat, annot=True)
plt.xlabel("Predictions")
plt.ylabel("Target")
plt.title("Confusion matrix for Test data")
plt.show()

#Let's calculate F1-Score, Precision Score and Recall Score for more clarification of the model
from sklearn.metrics import f1_score, precision_score, recall_score
test_f1 = f1_score(test_target, test_preds)
test_precision = precision_score(test_target, test_preds)
test_recall = recall_score(test_target, test_preds)

print("Accuracy on test set: ", (test_acc)*100, '%')
print('Test Set F1 Score: ', test_f1)
print("Test set Precision Score: ", test_precision)
print("Test Set Recall Score: ", test_recall)

# %% [markdown]
# ## Now the last step is Hyperparameter Tuning but as our model performs nicely on validation set (99.8% accuracy), it is not important to that

# %% [markdown]
# # Good, our model also performs well on test set with the accuracy of 99.95%. So it is one of the reliable model. Since we can use some more techniques for better predictions.

# %% [markdown]
# ## So it concludes our project on 'Credit Card Fraud Detection' using 'Random Forest Classification' algorithm.
# Thank you..!


