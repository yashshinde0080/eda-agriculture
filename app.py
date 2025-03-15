import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
df = pd.read_csv('agriculture_data.csv')

# Display the first 20 rows in the terminal
print("First 20 rows of the dataset:")
print(df.head(20).to_string(index=False))

# Clean the data
# Remove '%' and convert to numeric
df['Rainfall'] = df['Rainfall'].str.replace('%', '').astype(float)
df['Soil_Moisture'] = df['Soil_Moisture'].str.replace('%', '').astype(float)
df['Crop_Yield'] = df['Crop_Yield'].str.replace('%', '').astype(float)

# Remove 'mm' and convert to numeric
df['Irrigation'] = df['Irrigation'].str.replace(' mm', '').astype(float)

# Display first few rows in the desired format
print("Agriculture Data:")
print(df.head().to_string(index=False))

# Check for missing values and fill them (excluding the 'FieldID' column)
df.fillna(df.drop(columns=['FieldID']).mean(), inplace=True)

# Exploratory Data Analysis (EDA)
# Pairplot
sns.pairplot(df.drop(columns=['FieldID']), diag_kind='kde')
plt.suptitle('Pairplot of Agriculture Data', y=1.02)
plt.show()

# Correlation Matrix (excluding the 'FieldID' column)
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns=['FieldID']).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation')
plt.show()

# Feature Selection
X = df[['Rainfall', 'Irrigation', 'Soil_Moisture']]
y = df['Crop_Yield']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection & Training
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

best_model = None
best_score = -np.inf

for name, model in models.items():
    # Cross-validation for better evaluation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f'{name} Cross-Validation R² Scores: {cv_scores}')
    print(f'{name} Average Cross-Validation R² Score: {np.mean(cv_scores):.2f}')
    
    # Fit the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f'{name} Test R² Score: {score:.2f}')
    
    if score > best_score:
        best_score = score
        best_model = model

# Hyperparameter Tuning for XGBoost (if it's the best model)
if isinstance(best_model, XGBRegressor):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f'Best Parameters for XGBoost: {grid_search.best_params_}')

# Final Evaluation
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Best Model: {best_model}')
print(f'MAE: {mae:.2f}')
print(f'R² Score: {r2:.2f}')

# Stack Plot of Predictions
plt.figure(figsize=(10, 6))
plt.stackplot(range(len(y_test)), y_test, y_pred, labels=['Actual Yield', 'Predicted Yield'], alpha=0.6)
plt.xlabel("Fields")
plt.ylabel("Crop Yield (%)")
plt.title("Actual vs Predicted Crop Yield")
plt.legend(loc='upper left')
plt.show()

# Additional Visualizations
# Stem Plot for Rainfall vs Soil Moisture
plt.figure(figsize=(10, 6))
plt.stem(df['Rainfall'], df['Soil_Moisture'], linefmt='b-', markerfmt='bo', basefmt='r-')
plt.xlabel('Rainfall (%)')
plt.ylabel('Soil Moisture (%)')
plt.title('Rainfall vs Soil Moisture')
plt.grid(True)
plt.show()

# Stem Plot for Rainfall vs Crop Yield
plt.figure(figsize=(10, 6))
plt.stem(df['Rainfall'], df['Crop_Yield'], linefmt='b-', markerfmt='bo', basefmt='r-')
plt.xlabel('Rainfall (%)')
plt.ylabel('Crop Yield (%)')
plt.title('Rainfall vs Crop Yield')
plt.grid(True)
plt.show()

# Stack Plot for Rainfall, Soil Moisture, and Crop Yield
plt.figure(figsize=(10, 6))
plt.stackplot(df.index, df['Rainfall'], df['Soil_Moisture'], df['Crop_Yield'], labels=['Rainfall', 'Soil Moisture', 'Crop Yield'], alpha=0.6)
plt.xlabel("Field Index")
plt.ylabel("Percentage")
plt.title("Rainfall, Soil Moisture, and Crop Yield")
plt.legend(loc='upper left')
plt.show()

# Pie Chart for Distribution of Crop Yield
yield_bins = [0, 60, 70, 80, 90, 100]
yield_labels = ['Yield <60%', 'Yield 60-70%', 'Yield 70-80%', 'Yield 80-90%', 'Yield 90-100%']
df['Yield_Category'] = pd.cut(df['Crop_Yield'], bins=yield_bins, labels=yield_labels)
yield_distribution = df['Yield_Category'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(yield_distribution, labels=yield_labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Distribution of Crop Yield')
plt.show()

# Bar Plot for Irrigation vs Soil Moisture (Single Bar for Each Field)
plt.figure(figsize=(10, 6))
sns.barplot(x=df.index, y='Soil_Moisture', data=df, ci=None, palette='viridis')
plt.xlabel('Field Index')
plt.ylabel('Soil Moisture (%)')
plt.title('Field Index vs Soil Moisture')
plt.show()

# Bar Plot for Irrigation vs Crop Yield (Single Bar for Each Field)
plt.figure(figsize=(10, 6))
sns.barplot(x=df.index, y='Irrigation', data=df, ci=None, palette='viridis')
plt.xlabel('Field Index')
plt.ylabel('Irrigation (mm)')
plt.title('Field Index vs Irrigation')
plt.show()