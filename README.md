
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9],
    'Attendance': [60, 65, 70, 75, 80, 85, 90, 95],
    'Previous_Score': [50, 55, 60, 65, 70, 75, 80, 85],
    'Final_Score': [55, 58, 63, 68, 74, 79, 84, 89]
}

df = pd.DataFrame(data)

# Features and target
X = df[['Hours_Studied', 'Attendance', 'Previous_Score']]
y = df['Final_Score']

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction
new_student = [[6, 85, 72]]
prediction = model.predict(new_student)

print("Predicted Final Score:", round(prediction[0], 2))

# Scatter Plot
plt.scatter(df['Hours_Studied'], df['Final_Score'])
plt.xlabel('Hours Studied')
plt.ylabel('Final Score')
plt.title('Hours Studied vs Final Score')
plt.show()
