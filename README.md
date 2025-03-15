# eda-agriculture

This project analyzes agricultural data using machine learning techniques to predict crop yield based on factors like rainfall, irrigation, and soil moisture.

Data Processing:
- Removed unwanted characters (e.g., % and mm) and converted data to numerical format.
- Handled missing values by filling them with column means (excluding FieldID).

Exploratory Data Analysis (EDA):
- Used pair plots and correlation heatmaps to visualize relationships between features.
- Analyzed feature importance through various visualizations, including bar plots and stack plots.

Model Training & Evaluation:
- Used Random Forest and XGBoost regressors.
- Performed cross-validation to evaluate model performance.
- Selected the best model based on R² score.
- Applied hyperparameter tuning for XGBoost if it performed best.
- Evaluated the final model using MAE and R² score.

Visualization
- Stack plots to compare actual vs. predicted crop yield.
- Stem plots to analyze relationships between rainfall, soil moisture, and crop yield.
- Pie charts for crop yield distribution.
- Bar plots for irrigation and soil moisture analysis.

This analysis helps in understanding key agricultural factors influencing crop yield, improving decision-making for better farming strategies.
