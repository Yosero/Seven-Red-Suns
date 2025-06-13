import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sevenredsuns as srs

iris = sns.load_dataset('iris')
le = LabelEncoder()
iris['species'] = le.fit_transform(iris['species'])

data = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
results = iris['species'].values
num_classes = len(np.unique(results))

data = (data - data.mean(axis=0)) / data.std(axis=0)

data_train, data_test, results_train, results_test = train_test_split(
    data, results, test_size=0.2, random_state=1
)

input_size = data.shape[1]
hidden_size = 10
output_size = num_classes

nn = srs.Classificator(input_size, hidden_size, output_size, learning_rate=0.1)
nn.learn(data_train, results_train)

predictions = nn.predict(data_test)
correct_predictions = np.sum(predictions == results_test)
accuracy = correct_predictions / len(results_test)
print(f"\nAccuracy on test data: {accuracy}")
