import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sevenredsuns import BasicRegressor

flights = sns.load_dataset('flights')
flights = flights.dropna()
flights['month'] = flights['month'].map({'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                                         'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12})

X = flights[['year', 'month']].values
y = flights['passengers'].values.reshape(-1, 1)

scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)
input_size = X_train.shape[1]
hidden_size = 32
output_size = 1

model = BasicRegressor(input_size, hidden_size, output_size)
model.train(X_train, y_train, epochs=10000, learning_rate=0.01, batch_size=16)

predictions_scaled = model.predict(X_test)
predictions = scaler_y.inverse_transform(predictions_scaled)
y_test_orig = scaler_y.inverse_transform(y_test)

tolerance = 0.1
accuracy = np.mean(np.abs(predictions - y_test_orig) < tolerance * y_test_orig) * 100

plt.figure(figsize=(10, 6))
plt.scatter(y_test_orig, predictions, alpha=0.6)
plt.plot([y_test_orig.min(), y_test_orig.max()],
         [y_test_orig.min(), y_test_orig.max()], 'r--')
plt.title(f'Реальные vs Предсказанные значения\nТочность: {accuracy:.2f}% (в пределах {tolerance * 100}%)')
plt.xlabel('Реальное количество пассажиров')
plt.ylabel('Предсказанное количество пассажиров')
plt.grid(True)
plt.show()