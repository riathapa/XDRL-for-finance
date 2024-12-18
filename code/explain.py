import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('../explainability_data/cleaned_state_action_recopilation.csv',sep=',')

# Separate features and target
# Assuming the first 6 columns are actions
X = data.iloc[:, 6:]
y = data.iloc[:, :6]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree for multi-output regression
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Regression Report:")
print("MSE:", mean_squared_error(y_test, y_pred_dt))

# Feature Importances
importances = dt_model.feature_importances_
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values(by='importance', ascending=False)
print("Feature Importances:")
print(feature_importances)

# SHAP for multi-output regression
import shap

# Explain the model's predictions using SHAP
explainer = shap.TreeExplainer(dt_model)
shap_values = explainer.shap_values(X_test)

# Plot the SHAP values
shap.summary_plot(shap_values, X_test, plot_type='bar')
shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values, X_test, plot_type='violin')

# Save the SHAP values
shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
shap_values_df.to_csv('shap_values_dt.csv', index=False)

# Save the feature importances
feature_importances.to_csv('feature_importances_dt.csv', index=False)


# Train Random Forest for multi-output regression
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Regression Report:")
print("MSE:", mean_squared_error(y_test, y_pred_rf))

# Feature Importances
importances = rf_model.feature_importances_
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values(by='importance', ascending=False)
print("Feature Importances:")
print(feature_importances)

# SHAP for multi-output regression
import shap

# Explain the model's predictions using SHAP
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Plot the SHAP values
shap.summary_plot(shap_values, X_test, plot_type='bar')
shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values, X_test, plot_type='violin')

# Save the SHAP values
shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
shap_values_df.to_csv('shap_values_rf.csv', index=False)

# Save the feature importances
feature_importances.to_csv('feature_importances_rf.csv', index=False)
