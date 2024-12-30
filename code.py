import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, accuracy_score, r2_score, mean_squared_error, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree

port_data=pd.read_excel(r"C:\Users\desai\OneDrive\Desktop\Cleaned_student-por2.xlsx")
math_data=pd.read_excel(r"C:\Users\desai\OneDrive\Desktop\Cleaned_student-mat2.xlsx")

# Identify categorical columns
categorical_cols = port_data.select_dtypes(include=['object']).columns

# Apply Label Encoding
label_encoder = LabelEncoder()

for col in categorical_cols:
    port_data[col] = label_encoder.fit_transform(port_data[col])

# Identify categorical columns
categorical_cols1 = math_data.select_dtypes(include=['object']).columns

# Apply Label Encoding
label_encoder = LabelEncoder()

for col in categorical_cols1:
    math_data[col] = label_encoder.fit_transform(math_data[col])

data=pd.concat([math_data,port_data])

# corelation heat map
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Grades by gender
sns.boxplot(x='sex', y='G3', data=data)
plt.title("Final Grade by Gender")
plt.show()

# Grades by parental education level
sns.boxplot(x='Medu', y='G3', data=data)
plt.title("Final Grades by Mother's Education Level")
plt.show()

# Create binary target: 1 = pass (G3 >= 10), 0 = fail (G3 < 10)
data['pass_fail'] = data['G3'].apply(lambda x: 1 if x >= 10 else 0)
X = data.drop(columns=['G3', 'pass_fail'])
y_class = data['pass_fail']  # For classification
y_reg = data['G3']           # For regression

# For classification
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_class, test_size=0.3, random_state=42)

# For regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.3, random_state=42)


# Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_cls, y_train_cls)
y_pred_cls = clf.predict(X_test_cls)
# Random Forest Regressor
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train_reg, y_train_reg)
y_pred_reg = reg.predict(X_test_reg)

# Classification evaluation
print("Classification Report:")
print(classification_report(y_test_cls, y_pred_cls))
print(f"Accuracy: {accuracy_score(y_test_cls, y_pred_cls)}")

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test_cls, y_pred_cls), annot=True, cmap='Blues', fmt='d')
plt.title("Confusion Matrix")
plt.show()
# Recall measures how well the model identifies all actual positive cases.
#recision measures how many of the positive predictions (e.g., "Pass") are actually correct.
# This tells us the percentage of predictions that are correct
#The F1-score combines precision and recall into one metric, especially useful when the data is imbalanced.
 #support is just a count of how many examples there are for each class in the dataset. It helps us understand the size of each class and how the model performs on each class.
 #A confusion matrix is a tool used to evaluate the performance of a classification model It is a table that compares the model's predicted classifications with the actual (true) classifications, showing where the model got things right and where it made mistakes.


# Regression evaluation
print("Regression Metrics:")
print(f"R² Score: {r2_score(y_test_reg, y_pred_reg)}")
print(f"Mean Squared Error: {mean_squared_error(y_test_reg, y_pred_reg)}")

# Actual vs Predicted Plot
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
plt.title("Actual vs Predicted Grades")
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.show()
# Feature importance for classification
importances = pd.Series(clf.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Features for Classification")
plt.show()
# Feature importance for regression
importances_reg = pd.Series(reg.feature_importances_, index=X.columns)
importances_reg.nlargest(10).plot(kind='barh', color='green')
plt.title("Top 10 Features for Regression")
plt.show()
student_por=pd.read_excel(r"C:\Users\desai\OneDrive\Desktop\Cleaned_student-por2.xlsx")
student_mat=pd.read_excel(r"C:\Users\desai\OneDrive\Desktop\Cleaned_student-mat2.xlsx")
# Combine datasets
combined_data = pd.concat([student_mat, student_por])

# Define features and target variable
X = combined_data.drop(columns=['G3'])  # exclude final grade G3
y = combined_data['G3']  # Target variable

# One-hot encode categorical variables (One-hot encoding is a process used to convert categorical (non-numeric) 
#(data into a numerical format that machine learning models can understand.)
X_encoded = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Decision Tree Regressor
decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(X_train, y_train)

# Predictions and evaluation
y_pred = decision_tree.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
# Visualize Feature Importances
import matplotlib.pyplot as plt
import numpy as np

# Get feature importances and sort them
feature_importances = decision_tree.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]
sorted_features = [X_encoded.columns[i] for i in sorted_idx]

# Ploting the feature importances
plt.figure(figsize=(10, 6))
plt.bar(sorted_features[:10], feature_importances[sorted_idx][:10])
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Top 10 Feature Importances")
plt.xticks(rotation=45)
plt.show()

# the decision tree
plt.figure(figsize=(15, 10))
plot_tree(
    decision_tree,
    feature_names=X_encoded.columns,
    filled=True,
    rounded=True,
    max_depth=3  
)
plt.title("Decision Tree Visualization")
plt.show()


# Create a scatter plot for Actual vs. Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Values (G3)")
plt.ylabel("Predicted Values (G3)")
plt.title("Actual vs. Predicted Scatter Plot")
plt.axline((0, 0), slope=1, color="red", linestyle="--", linewidth=2, label="Perfect Prediction Line")  
plt.legend()
plt.show()

