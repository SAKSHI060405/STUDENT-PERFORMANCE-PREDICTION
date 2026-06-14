import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("student_data.csv")

# Features and target
X = data[['study_hours', 'attendance', 'previous_score']]
y = data['final_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Accuracy
error = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", error)

# Example prediction
new_student = [[6, 90, 75]]
predicted_score = model.predict(new_student)

print("Predicted Final Score:", predicted_score[0])

# Graph
plt.scatter(y_test, predictions)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Student Performance Prediction")
plt.show()