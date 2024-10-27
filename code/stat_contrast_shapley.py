import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import shap
import numpy as np
import matplotlib.pyplot as plt

# Load the merged CSV file
file_path = 'cleaned_state_action_10000.csv'
df = pd.read_csv(file_path)

# Separate the features (states) and target (actions)
state_columns = [col for col in df.columns if col.startswith('State')]
action_columns = [col for col in df.columns if col.startswith('Action')]

X = df[state_columns]
y = df[action_columns]


#Meter esto dentro de un bucle 50 veces con distinta semilla. Recoger los shapley. Hacer un contraste con respecto a una muestra de uno.
shap_avg_values = []

# Split the data into training and testing sets
samples = 50
for iteration in range(samples):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) #Fixed.

    # Train the model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=iteration)
    rf_model.fit(X_train, y_train)

    # Improve the model
    # Tune the hyperparameters of the Random Forest model
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error

    '''
    # Define the hyperparameters
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Instantiate the GridSearchCV object
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

    # Fit the GridSearchCV object
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:")
    print(best_params)
    '''
    
    # Get the best model
    #best_rf_model = grid_search.best_estimator_
    best_rf_model = rf_model

    # Evaluate the best model
    y_pred_best_rf = best_rf_model.predict(X_test)
    print("Best Random Forest Regression Report:")
    print("MSE:", mean_squared_error(y_test, y_pred_best_rf))


    explainer = shap.TreeExplainer(best_rf_model)

    # Calculate SHAP values
    # 6 acciones, basta quedarse con 1 de las acciones, vamos a quedarnos con AAPL.
    # 24 variables en X, eso explica el segundo eje. 
    # 100 filas en x_test, eso explica el primer eje. 
    # Cada elemento en test tiene 1 valor de shapley. Basta quedarse con uno.
    # Nos vamos a quedar con todos los shapley values 
    shap_values = explainer.shap_values(X_test) #100,24,6
    for index_variable in range(X_train.shape[1]):
        shap_values_variable = np.mean(shap_values, axis=0)[:,1]
        shap_avg_values.append({'Shapley' : shap_values_variable[index_variable], 'Variable' : index_variable, 'Seed' : iteration})

    # print shap_values shape
    print("SHAP Values Shape:")
    print(shap_values.shape)

#Now we have all the values and we can do the statistical hypothesis testing.
#Mixed-Effects Model: Use a linear mixed-effects model where:
#Fixed Effects: Variables (features).
#Random Effects: Seeds
import statsmodels.formula.api as smf
df = pd.DataFrame(shap_avg_values)

import pdb; pdb.set_trace();

# Since the Shapley values are highly consistent across variables and seeds, focus on descriptive statistics to showcase this consistency.
overall_mean = df['Shapley'].mean()
overall_std = df['Shapley'].std()
print(f"Overall Mean Shapley Value: {overall_mean}")
print(f"Overall Standard Deviation: {overall_std}")

# Highlight that the variances are extremely low, indicating high consistency.
variance_by_variable = df.groupby('Variable')['Shapley'].var()
print("Variance of Shapley values by Variable:")
print(variance_by_variable)


import seaborn as sns
import matplotlib.pyplot as plt

#Histogram of Shapley Values:
sns.histplot(df['Shapley'], kde=True)
plt.title('Distribution of Shapley Values')
plt.xlabel('Shapley Value')
plt.ylabel('Frequency')
plt.show()

#by variable
plt.figure(figsize=(12, 6))
sns.boxplot(x='Variable', y='Shapley', data=df)
plt.xticks(rotation=90)
plt.title('Shapley Values by Variable')
plt.show()

#by seed
plt.figure(figsize=(12, 6))
sns.boxplot(x='Seed', y='Shapley', data=df)
plt.xticks(rotation=90)
plt.title('Shapley Values by Seed')
plt.show()


# Fit the mixed-effects model
model = smf.mixedlm("Shapley ~ Variable", df, groups=df["Seed"])
result = model.fit()

# Print the summary of the model
print(result.summary())
