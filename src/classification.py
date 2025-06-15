import numpy as np

class BasicClassificator:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, input_data):
        self.hidden_layer_input = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = self.sigmoid(self.output_layer_input)
        return self.output_layer_output

    def backward(self, input_data, target_data, output_data):
        output_error = target_data - output_data
        output_delta = output_error * self.sigmoid_derivative(output_data)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)

        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += input_data.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def learn(self, data, results, epochs=100):
        for epoch in range(epochs):
            for i in range(len(data)):
                input_data = data[i:i+1]
                target_data = np.zeros(self.output_size)
                target_data[int(results[i])] = 1
                target_data = target_data.reshape(1, -1)
                output_data = self.forward(input_data)
                self.backward(input_data, target_data, output_data)

            if (epoch + 1) % 100 == 0:
                predictions = self.predict(data)
                correct = np.sum(predictions == results)
                accuracy = correct / len(data)

    def predict(self, data):
        output = self.forward(data)
        return np.argmax(output, axis=1)