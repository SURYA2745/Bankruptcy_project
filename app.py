# %%
#what is bankruptacy?
#when someone (or a company) is unable to repay their debts, they can file for bankruptcy. 
#This gives them a fresh start by either reducing or eliminating their debts. However, it may involve selling assets or restructuring how the debt is paid.
#Ex:Toys "R" Us (2017):
#What Happened: Toys "R" Us, a popular toy store chain, filed for bankruptcy because it had accumulated too much debt and couldn't compete with online retailers like Amazon.
#Result: The company closed many of its stores and had to restructure its operations.

#load the dataset
import pandas as pd
import numpy as np
data=pd.read_excel("Bankruptcy.xlsx")

# %%
data

# %%
data.shape

# %%
data.columns

# %%
data.describe()

# %%
#check for missing values
data.isnull().sum()

# %%
#check data types
data.dtypes

# %%
#Histogram plot
import matplotlib.pyplot as plt
import seaborn as sns
features = ['industrial_risk', 'management_risk', 'financial_flexibility', 
            'credibility', 'competitiveness', 'operating_risk']

# Set up the plot
plt.figure(figsize=(10, 8))

# Loop through each feature and plot the histogram
for i, feature in enumerate(features, 1):
    plt.subplot(2, 3, i) 
    plt.hist(data[feature], bins=3, edgecolor='black')  
    plt.title(f'{feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

# %%
#Box plot:A boxplot is useful for visualizing the distribution of data and identifying outliers. 
# Loop through each feature and plot the boxplot
for i, feature in enumerate(features, 1):
    plt.subplot(2, 3, i)  # 2 rows, 3 columns
    sns.boxplot(data=data[feature])
    plt.title(f'{feature} Boxplot')
    plt.xlabel(feature)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

# %%
# Countplot
sns.countplot(x="class", data=data)
plt.title("Distribution of Bankruptcy vs Non-Bankruptcy")
plt.show()

# %%
class_counts = data['class'].value_counts()
# Print the counts
print(class_counts)

# %%
#The plot contains two bars (since class is a binary variable, with values 0 and 1):
#One bar will represent the number of non-bankrupt companies (where class = 0).
#The other bar will represent the number of bankrupt companies (where class = 1).

#In this case Non bankruptacy is higher than bankruptacy it means that more companies in the dataset did not go bankrupt.

# %%
# Pair plot to explore feature relationships
sns.pairplot(data)
plt.show()

# %%
numeric_data = data.select_dtypes(include=['number'])

# %%
# Correlation matrix to analyze relationships between features
# Compute the correlation matrix
correlation_matrix = numeric_data.corr()

# Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()

# %%
#Red colors indicate strong positive correlations (close to 1), meaning that as one feature increases, the other increases too.
#Blue colors indicate strong negative correlations (close to -1), meaning that as one feature increases, the other decreases.
#White or near-zero color indicates weak or no correlation (close to 0), meaning there is little to no linear relationship between the features.

#The diagonal of the heatmap shows a correlation of 1, as each feature is perfectly correlated with itself.


# %% [markdown]
# Model Building:
# Splitting Data: Split data into training and testing sets.
# Model Selection: Choose an appropriate classification algorithm

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
#feature scaling: standardizing the features
scaler=StandardScaler()
scaled_features=scaler.fit_transform(data.drop(columns='class'))
#convert the scaled featutes back to DataFrame
scaled_data=pd.DataFrame(scaled_features, columns=data.columns[:-1])
#add the target variable back
scaled_data['class']=data['class']
#display the scaled result
print(scaled_data.head())

# %% [markdown]
# industrial_risk: This represents the level of risk in the business industry.
# 
# The values are scaled between -1 and 1, indicating different risk levels.
# A value close to 1 indicates high risk.
# A value close to -1 indicates low risk.
# A value around 0 suggests a medium level of risk.
# in all case, Positive values represent higher risk, and negative values represent lower risk.

# %% [markdown]
# For example: industrial_risk = -0.043827
# management_risk = 0.941732
# financial_flexibility = -0.938172
# credibility = -1.132941
# competitiveness = -1.08231
# operating_risk = -0.161400
# class = bank
# 
# This row represents a company with low industrial risk (-0.043827), but it has a high management risk (0.941732) and very low financial flexibility (-0.938172). The company is also not credible (-1.132941) and has low competitiveness (-1.08231). The operating risk is relatively low (-0.161400). This company went bankrupt according to the class label.ruptcy

# %%
#splitting data into training and testing sets
#befpre training the model we need to split the data into a training set and a testing set.

#split data into features(x) and target(y)
X=scaled_data.drop(columns='class')
y=scaled_data['class']

# %%
#split the data into training(80%) & tetsing set(20%)
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42)

# %%
print(f"Training data size: {X_train.shape}")
print(f"Tetsing data size: {X_test.shape}")

# %% [markdown]
# X_train: This is the data that the model will use to learn the patterns. The shape of X_train tells us how many samples (or rows) and how many features (or columns) we have in the training set.
# X_test: This is the data we will use to test the model's performance. The shape of X_test tells us how many samples and features we have in the testing set.
# 
# Training data size: (200, 6)-----> 200 rows 6 columns
# Tetsing data size: (50, 6-----> 50 rows 6 columns)

# %%
#Model building and training
#in this step we train different classification models, such as logistic regression, decision trees and radom forest to predict bankruptacy
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# %%
#intialize the models
log_reg=LogisticRegression()
dt=DecisionTreeClassifier(random_state=42)
rf=RandomForestClassifier(random_state=42)
#by setting a random_state (e.g., random_state=42), 
#every time you run the model, it will follow the same path in building the tree, ensuring that your results are consistent.

# %%
#train the model
log_reg.fit(X_train, y_train)

# %%
# Convert y_test to numeric labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_pred_log_reg = log_reg.predict(X_test)
y_pred_log_reg_numeric = label_encoder.fit_transform(y_pred_log_reg)
y_test_numeric = label_encoder.fit_transform(y_test)

# %%
dt.fit(X_train, y_train)

# %%
rf.fit(X_train, y_train)

# %%
#make predictions
y_pred_log_reg=log_reg.predict(X_test)
y_pred_dt=dt.predict(X_test)
y_pred_rf=rf.predict(X_test)

# %%
# Evaluate the model
print("Logistic Regression Evaluation:")
print(classification_report(y_test_numeric, y_pred_log_reg_numeric))
print(confusion_matrix(y_test_numeric, y_pred_log_reg_numeric))
y_pred_log_reg_prob = log_reg.predict_proba(X_test)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test_numeric, y_pred_log_reg_prob)}\n")

# %% [markdown]
# a. Precision:
# Precision for class 0 (non-bankruptcy) = 1.00: This means that when the model predicted non-bankruptcy, it was correct 100% of the time.
# Precision for class 1 (bankruptcy) = 1.00: This means that when the model predicted bankruptcy, it was correct 100% of the time.
# 
# b. Recall:
# Recall for class 0 (non-bankruptcy) = 1.00: This means that the model correctly identified all the non-bankrupt companies.
# Recall for class 1 (bankruptcy) = 1.00: This means that the model correctly identified all the bankrupt companies.
# 
# c. F1-Score:
# F1-score for both classes = 1.00: The F1-score is the harmonic mean of precision and recall. Since both precision and recall are 1.00 for both classes, the F1-score is also perfect (1.00), indicating a great balance between precision and recall.
# 
# Confusion matrix:
# [[21  0]   21 non-bankrupt companies correctly classified, 0 misclassified as bankrupt
#  [ 0 29]] # 29 bankrupt companies correctly classified, 0 misclassified as non-bankrup
# 
#  The ROC-AUC score of 1.00 is the best possible score, indicating that the model is perfectly classifying both bankruptcy and non-bankruptcy companies and has no confusion between the two classes.t

# %%
label_encoder = LabelEncoder()
y_test_numeric = label_encoder.fit_transform(y_test)
y_pred_dt_numeric = label_encoder.transform(y_pred_dt) 
y_pred_rf_numeric = label_encoder.transform(y_pred_rf) 

# %%
print("Decision Tree Evaluation:")
print(classification_report(y_test_numeric, y_pred_dt_numeric))
print(confusion_matrix(y_test_numeric, y_pred_dt_numeric))
print(f"ROC-AUC: {roc_auc_score(y_test_numeric, y_pred_dt_numeric)}\n")

# %% [markdown]
# The Decision Tree model performs very well, achieving 98% accuracy, with strong precision, recall, and F1-scores for both classes (bankruptcy and non-bankruptcy).
# 
# confusion matrix: 
# [[21  0]     --> 21 correct non-bankruptcy predictions (0)
#  [ 1 28]]    --> 1 incorrect bankruptcy prediction (0 predicted as bankruptcy) and 28 correct bankruptcy predictions (1
#  There’s a small error: 1 case of bankruptcy was misclassified as non-bankruptcy, but overall, the model is very good
# 
# The ROC-AUC score of 0.98 indicates the model's excellent ability to differentiate between bankrupt and non-bankrupt companies..

# %%
print("Random Forest Evaluation:")
print(classification_report(y_test_numeric, y_pred_rf_numeric))
print(confusion_matrix(y_test_numeric, y_pred_rf_numeric))
print(f"ROC-AUC: {roc_auc_score(y_test_numeric, y_pred_rf_numeric)}\n")

# %% [markdown]
# The Random Forest model performed perfectly, achieving 100% accuracy, with 1.00 precision, 1.00 recall, and 1.00 F1-scores for both classes (bankruptcy and non-bankruptcy).
# 
# The ROC-AUC score of 1.0 confirms that the model perfectly discriminates between bankrupt and non-bankrupt companies.
# 
# The confusion matrix shows that all the predictions were correct (no false positives or false negatives).
# 
# This indicates that the Random Forest model is extremely reliable for predicting bankruptcy in this dataset. It is able to classify every instance accurately, making it a highly effective model for this task.

# %%
import pickle
from sklearn.ensemble import RandomForestClassifier


# %%
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# %%
with open("bankruptcy_model.pkl", "wb") as file:
    pickle.dump(model, file)

# %%
#pip install streamlit


# %%
import streamlit as st
import pickle
import numpy as np


# %%
with open("bankruptcy_model.pkl", "rb") as file:
    model = pickle.load(file)

# %%
st.title("Bankruptcy Prediction App")

# %%
st.write("Enter the company's financial details to predict bankruptcy risk.")

# %%
industrial_risk = st.selectbox("Industrial Risk", [0, 0.5, 1])
management_risk = st.selectbox("Management Risk", [0, 0.5, 1])
financial_flexibility = st.selectbox("Financial Flexibility", [0, 0.5, 1])
credibility = st.selectbox("Credibility", [0, 0.5, 1])
competitiveness = st.selectbox("Competitiveness", [0, 0.5, 1])
operating_risk = st.selectbox("Operating Risk", [0, 0.5, 1])


# %%
if st.button("Predict"):
    features = np.array([[industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]])
    prediction = model.predict(features)

    # Output result
    result = "Bankrupt" if prediction[0] == 1 else "Non-Bankrupt"
    st.write(f"### Prediction: {result}")

# %%


# %%



