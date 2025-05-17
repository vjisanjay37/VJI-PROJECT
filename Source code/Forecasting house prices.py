
# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Step 1: Data Preparation (Synthetic Data Generation)
np.random.seed(42)

# Generating synthetic data
n_samples = 100
size = np.random.randint(500, 3500, size=n_samples)  # Random house sizes between 500 and 3500 sq ft
rooms = np.random.randint(1, 10, size=n_samples)  # Random number of rooms (1 to 10 rooms)
age = np.random.randint(0, 100, size=n_samples)  # Random house age (0 to 100 years)

# Target variable: House price (in thousands of dollars)
price = 50 + (size * 0.15) + (rooms * 10) - (age * 0.1) + np.random.normal(0, 20000, size=n_samples)

# Creating a DataFrame
data = pd.DataFrame({
    'Size': size,
    'Rooms': rooms,
    'Age': age,
    'Price': price
})

# Step 2: Data Exploration
print("First few rows of the data:")
print(data.head())

# Visualizing data distributions
plt.figure(figsize=(12, 6))

# Scatter plot for Size vs Price
plt.subplot(1, 3, 1)
plt.scatter(data['Size'], data['Price'], color='blue')
plt.title('Size vs Price')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price ($1000)')

# Scatter plot for Rooms vs Price
plt.subplot(1, 3, 2)
plt.scatter(data['Rooms'], data['Price'], color='green')
plt.title('Rooms vs Price')
plt.xlabel('Number of Rooms')
plt.ylabel('Price ($1000)')

# Scatter plot for Age vs Price
plt.subplot(1, 3, 3)
plt.scatter(data['Age'], data['Price'], color='red')
plt.title('Age vs Price')
plt.xlabel('Age of House (years)')
plt.ylabel('Price ($1000)')

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', cbar=True, linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Step 3: Preparing the data for modeling
X = data[['Size', 'Rooms', 'Age']]  # Features
y = data['Price']  # Target variable

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Initialization
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(random_state=42)
}

# Step 5: Model Training and Evaluation
results = {}

for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[model_name] = {'MSE': mse, 'R2 Score': r2}
    
    # Plot predictions vs actual values (optional)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)  # Ideal line
    plt.title(f'{model_name} Predictions vs Actual Prices')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.grid(True)
    plt.show()

# Step 6: Display Results
print("\nModel Performance:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  Mean Squared Error (MSE): {metrics['MSE']:.2f}")
    print(f"  R² Score: {metrics['R2 Score']:.2f}")

# Step 7: Selecting the Best Model
best_model_name = max(results, key=lambda x: results[x]['R2 Score'])
best_model = models[best_model_name]

print(f"\nThe best model is {best_model_name} with an R² Score of {results[best_model_name]['R2 Score']:.2f}")
