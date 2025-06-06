# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from IPython.display import display, HTML
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# %%
df = pd.read_csv("https://raw.githubusercontent.com/DS3001/linearRegression/refs/heads/main/data/Q1_clean.csv")
# make all the column names lower case
df.columns = df.columns.str.lower()

# Print the column names to check if 'neighbourhood' is present
print(df.columns)

# %%
df1 = pd.get_dummies(df, columns=["neighbourhood ",'property type','room type'], drop_first=True)
# %%
# %%
# Define the features and target variable
X = df1.drop(columns=[col for col in df1.columns if col.startswith(('room', 'price', 'property'))])
y = df1['price']

# %%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

#%%
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# Calculate R^2 score
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2}")


# Calculate RMSE
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Print the coefficients
print("Coefficients:", model.coef_)
# Create a series with the column names and the coefficients
coefficients = pd.Series(model.coef_, index=X.columns)
# Print the series
print(coefficients)


################################## Q1 Model 2 #########################################
# %%
# Next Model
X1 = df1.drop(columns=[col for col in df1.columns if col.startswith(('room', 'price'))])
y = df1['price']
# %%
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=44)

#%%
# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

#%%
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# Calculate R^2 score
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2}")


# Calculate RMSE
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# %%
# Print the coefficients
print("Coefficients:", model.coef_)
# Create a series with the column names and the coefficients
coefficients = pd.Series(model.coef_, index=X1.columns)
# Print the series
print(coefficients)


############################ Q2 Data Prep #################################################

#%%
# Load the data
cars_df = pd.read_csv('data/cars_hw.csv')

# Clean the data

#%%
# Check for outliers in numeric columns
numeric_cols = cars_df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    sns.boxplot(x=cars_df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

#%%

# Remove outliers
cars_df = cars_df[(cars_df['Price'] > cars_df['Price'].quantile(0.01)) & 
          (cars_df['Price'] < cars_df['Price'].quantile(0.99))]

#%%
# Apply log transformation to skewed variables
cars_df['Price'] = np.log(cars_df['Price'])

#%%
# Summarize the Price variable
print(cars_df['Price'].describe())

# Create a kernel density plot
sns.kdeplot(cars_df['Price'])
plt.title('Kernel Density Plot of Price')
plt.show()

# Summarize prices by brand
price_summary_by_brand = cars_df.groupby('Make')['Price'].describe()
print(price_summary_by_brand)

# Create a grouped kernel density plot by Make
sns.kdeplot(data=cars_df, x='Price', hue='Make', common_norm=False)
plt.title('Kernel Density Plot of Price by Make')
plt.show()

#%%
# Split the data into training and testing sets
X = cars_df.select_dtypes(include=[np.number]).drop(columns=['Price'])
y = cars_df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

#%%

############################# Model 1: Regress price on numeric variables alone ####################################
model1 = LinearRegression()
model1.fit(X_train, y_train)
y_train_pred1 = model1.predict(X_train)
y_test_pred1 = model1.predict(X_test)
r2_train1 = r2_score(y_train, y_train_pred1)
rmse_train1 = np.sqrt(mean_squared_error(y_train, y_train_pred1))
r2_test1 = r2_score(y_test, y_test_pred1)
rmse_test1 = np.sqrt(mean_squared_error(y_test, y_test_pred1))
print(f"Model 1 - R^2 on training set: {r2_train1}, RMSE on training set: {rmse_train1}")
print(f"Model 1 - R^2 on test set: {r2_test1}, RMSE on test set: {rmse_test1}")


#%%
#### Model 2: Regress price on one-hot encoded categorical variables alone ###############
X_cat = pd.get_dummies(cars_df.select_dtypes(include=['object']), drop_first=True)
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y, test_size=0.2, random_state=42)
model2 = LinearRegression()
model2.fit(X_train_cat, y_train_cat)
y_test_pred2 = model2.predict(X_test_cat)
r2_test2 = r2_score(y_test_cat, y_test_pred2)
rmse_test2 = np.sqrt(mean_squared_error(y_test_cat, y_test_pred2))
print(f"Model 2 - R^2 on test set: {r2_test2}, RMSE on test set: {rmse_test2}")


#%%
####### Model 3: Combine all regressors from previous two models ##################
X_combined = pd.concat([X, X_cat], axis=1)
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y, test_size=0.2, random_state=40)
model3 = LinearRegression()
model3.fit(X_train_combined, y_train_combined)
y_test_pred3 = model3.predict(X_test_combined)
r2_test3 = r2_score(y_test_combined, y_test_pred3)
rmse_test3 = np.sqrt(mean_squared_error(y_test_combined, y_test_pred3))
print(f"Model 3 - R^2 on test set: {r2_test3}, RMSE on test set: {rmse_test3}")

#%%

################### Model 4: Polynomial Features ####################################
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2, random_state=44)
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train_poly)
y_test_pred_poly = model_poly.predict(X_test_poly)
r2_test_poly = r2_score(y_test_poly, y_test_pred_poly)
rmse_test_poly = np.sqrt(mean_squared_error(y_test_poly, y_test_pred_poly))
print(f"Polynomial Model - R^2 on test set: {r2_test_poly}, RMSE on test set: {rmse_test_poly}")


#%%
# Plot predicted vs true values for the best model
plt.scatter(y_test_combined, y_test_pred3)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs True Values')
plt.show()

#%%
# Plot predicted vs true values for the polynomial model
plt.scatter(y_test_poly, y_test_pred_poly)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Polynomial Model: Predicted vs True Values')
plt.show()


#%%
# Compute residuals and create a kernel density plot
residuals = y_test - y_test_pred3
sns.kdeplot(residuals)
plt.title('Kernel Density Plot of Residuals')
plt.show()

# %%
# Create a dictionary with the results
results = {
    "Model": ["Model 1", "Model 2", "Model 3", "Polynomial Model"],
    "R^2 (Test Set)": [r2_test1, r2_test2, r2_test3, r2_test_poly],
    "RMSE (Test Set)": [rmse_test1, rmse_test2, rmse_test3, rmse_test_poly]
}

# Create a DataFrame from the results dictionary
results_df = pd.DataFrame(results)

# Convert the DataFrame to an HTML table
html_table = results_df.to_html(index=False)

# Display the HTML table
display(HTML(html_table))

