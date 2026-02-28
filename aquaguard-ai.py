import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, 
                             roc_auc_score, RocCurveDisplay)
# Load and inspect dataset
df = pd.read_csv("WaterLeak.csv")
l = list(df['Leakage_Flag'].value_counts())
circle = [l[1] / sum(l) * 100, l[0] / sum(l) * 100]
colors = ['#F3ED13','#451FA4']
# --- Graph 1: Pie Chart ---
plt.figure(figsize=(8, 6))
plt.pie(circle, labels=['Leakage_Flag', 'Not Leakage_Flag'], autopct='%1.1f%%', colors=colors)
plt.title('Leakage_Flag %')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='Leakage_Flag', data=df, palette=colors)
plt.title('Cases of Leakage')
plt.show()

# Chi-Square test for categorical variables
df_cat = df.copy()
le = LabelEncoder()
cat_cols = ['Zone', 'Block', 'Pipe', 'Location_Code']
for col in cat_cols:
    df_cat[col] = le.fit_transform(df_cat[col])

X_fs = df_cat[cat_cols]
y_fs = df_cat['Leakage_Flag']
best_features = SelectKBest(score_func=chi2, k='all')
fit = best_features.fit(X_fs, y_fs)
featureScores = pd.DataFrame(data=fit.scores_, index=list(X_fs.columns), columns=['Chi Squared Score'])

plt.figure(figsize=(6, 4))
sns.heatmap(featureScores.sort_values(by='Chi Squared Score', ascending=False), annot=True, cmap='YlGnBu')
plt.title('Selection of Categorical Features')
plt.show()

#Anova test for continuous variables
from sklearn.feature_selection import f_classif

# give numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(['Leakage_Flag'], errors='ignore')

#  Run anova through f_classif
f_values, p_values = f_classif(df[num_cols], df['Leakage_Flag'])

anova_scores = pd.DataFrame({'Feature': num_cols, 'ANOVA F-Score': f_values}).set_index('Feature')

# show graph
plt.figure(figsize=(6, 4))
sns.heatmap(anova_scores.sort_values(by='ANOVA F-Score', ascending=False), annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('ANOVA Score: Numerical Feature Selection')
plt.show()

# Data Distribution plots
# Distribution plots for all numerical variables
df.hist(figsize=(12,10))
plt.tight_layout()
plt.show()

#Correlation Heatmap
plt.figure(figsize=(10,8))
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# AI Model setup
df_modelX = df.drop(['Location_Code', 'Leakage_Flag'], axis=1)
for col in ['Zone', 'Block', 'Pipe']:
    df_modelX[col] = le.fit_transform(df_modelX[col])
X_train, X_test, y_train, y_test = train_test_split(df_modelX, df['Leakage_Flag'], test_size=0.2, random_state=1, stratify=df['Leakage_Flag'])

def model_eval(classifier, name):
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    cv_score = cross_val_score(classifier, X_train, y_train, cv=cv).mean()
    
    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {accuracy_score(y_test, prediction):.2%}")
    print(f"Cross Val Score: {cv_score:.2%}")
    print(f"ROC_AUC: {roc_auc_score(y_test, classifier.predict_proba(X_test)[:,1]):.2%}")
    
    RocCurveDisplay.from_estimator(classifier, X_test, y_test)
    plt.title(f'ROC AUC - {name}')
    plt.show()
    print(classification_report(y_test, prediction))

# Logistic Regression
classifier_lr = LogisticRegression(max_iter=1000)
model_eval(classifier_lr, "Logistic Regression")

# Random Forest
classifier_rf = RandomForestClassifier(max_depth=4, random_state=0)
model_eval(classifier_rf, "Random Forest")

importances = pd.DataFrame({'Attribute': X_train.columns, 'Importance': classifier_rf.feature_importances_})
importances = importances.sort_values(by='Importance', ascending=False)
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('RF Feature Importance')
plt.xticks(rotation='vertical')
plt.show()

# Confusion Matrix
#  predictions
y_pred = classifier_lr.predict(X_test)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize 
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Leak', 'Leak'], 
            yticklabels=['No Leak', 'Leak'])

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# double check
print("Confusion Matrix Array:")
print(cm)

# Saving Ramdom forest data output into HTML 
import joblib


# 1. Load data
df = pd.read_csv("WaterLeak.csv")

# 2. Data Cleaning
drop_list = ['Leakage_Flag', 'Location_Code', 'Latitude', 'Longitude', 'Unnamed: 0']
df_modelX = df.drop(columns=[c for c in drop_list if c in df.columns])

# Categories
le = LabelEncoder()
for col in ['Zone', 'Block', 'Pipe']:
    df_modelX[col] = le.fit_transform(df_modelX[col].astype(str))

# Train Random Forest Model
# We use max_depth=5 to ensure the model generalizes well to new data
final_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
final_model.fit(df_modelX, df['Leakage_Flag'])

 # Predictions
df_predictions = df.copy()
df_predictions['Leak_Probability'] = final_model.predict_proba(df_modelX)[:,1]
df_predictions['Predicted_Leak'] = final_model.predict(df_modelX)

# Risk Classification

def classify_risk(prob):
    if prob >= 0.75: return "High"
    elif prob >= 0.40: return "Medium"
    else: return "Low"

df_predictions['Risk_Level'] = df_predictions['Leak_Probability'].apply(classify_risk)
df_predictions = df_predictions.sort_values(by='Leak_Probability', ascending=False)
df_predictions['Priority_Rank'] = range(1, len(df_predictions)+1)

# Save Outputs
df_predictions.to_csv("predicted_next_month.csv", index=False)
joblib.dump(final_model, "rf_leak_model.pkl")
print(" Random Forest model and Dashboard data saved.")
