#importing required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#path variables
path1_train = "/home/megh-ops/Documents/data/ds1_train.csv"
path1_test = "/home/megh-ops/Documents/data/ds1_test.csv"
path2_train = "/home/megh-ops/Documents/data/ds2_train.csv"
path2_test = "/home/megh-ops/Documents/data/ds2_test.csv"

#methods to calculate weights and predict the target
#has member variables threshold and weights
class Predictor:

    def __init__(self, threshold):
        self.threshold = threshold

    #sigmoid function
    def sigmoid(self, x, weight):
        z = np.dot(x, weight)
        return np.reshape(1/ (1 + np.exp(-z)), (x.shape[0], 1))

    #calculating the loss function
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1-h)).mean()

    #calculating the gradients
    def gradient_descent(self, x, h, y):
        return np.dot(x.T, (h - y)) / y.shape[0]

    #method to update the weights and biases and creates an array containing the loss values
    def learn(self, trainingkit):
        weights = np.zeros(trainingkit.x.shape[1], dtype = float)
        self.weight = np.reshape(weights, (trainingkit.x.shape[1], 1))
        loss_list = []
        for i in range(trainingkit.iter):
            sigma = self.sigmoid(trainingkit.x, self.weight)
            loss_list.append(self.loss(sigma, trainingkit.y))
            dW = self.gradient_descent(trainingkit.x, sigma, trainingkit.y)
            self.weight -= trainingkit.lr * dW
        self.loss_list = loss_list

    #function to plot the loss against number of iterations
    def plot_loss(self):
        plt.plot(self.loss_list)
        plt.ylabel('Loss function')
        plt.xlabel('Iterations')
        plt.title('Training Set Loss Plot')
        plt.show()

    #computes expected target values based on the weights and bias calculated
    def predict(self, testkit):
        result = self.sigmoid(testkit.x, self.weight)
        result = result >= self.threshold
        y_pred = np.zeros(result.shape[0])
        for i in range(len(y_pred)):
            if result[i] == True:
                y_pred[i] = 1
            else:
                continue
        return np.reshape(y_pred, (testkit.y.shape[0], 1))


#initialises the features and expected output and stores as numpy arrays
#called by functions to calculate the parameters of the model
#has member variables x(features) ,y(target), number of iterations and learning rate hyperparameter
class TrainingKit:
    def __init__(self, path, iteration, lr):
        dframe = pd.read_csv(path)
        dframe_x = dframe[['x_1', 'x_2']]
        x1 = np.array(dframe_x, dtype = float)
        intercept = np.ones((x1.shape[0], 1))
        self.x = np.concatenate((intercept, x1), axis = 1)
        dframe_y = dframe[['y']]
        self.y = np.array(dframe_y, dtype = float)
        self.iter = iteration
        self.lr = lr

#a child class of trainingkit
#only used for predicting the target and not to calculate the weights of the model
class Testkit(TrainingKit):
    pass

#setting the number of iterations for the weights to converge and learning rate
num_iterations = 200000
lr = 0.001
threshold = 0.5
predictor = Predictor(threshold)

#creating instances of training and test data
trainkit1 = TrainingKit(path1_train, num_iterations, lr)
testkit1 = Testkit(path1_test, num_iterations, lr)

trainkit2 = TrainingKit(path2_train, num_iterations, lr)
testkit2 = Testkit(path2_test, num_iterations, lr)

#learn method calculates the parameters of the model using training data
predictor.learn(trainkit1)
predictor.plot_loss()
#predict method returns an array of expected values which is then compared to the expected output to compute the accuracy
y_pred = predictor.predict(trainkit1)
print('Accuracy of training dataset1 is {}'.format(sum(y_pred == trainkit1.y) / trainkit1.y.shape[0]))
y_pred = predictor.predict(testkit1)
print('Accuracy of test dataset1 is {}'.format(sum(y_pred == testkit1.y) / testkit1.y.shape[0]))

num_iterations = 100000

predictor.learn(trainkit2)
predictor.plot_loss()
y_pred = predictor.predict(trainkit2)
print('Accuracy of training dataset2 is {}'.format(sum(y_pred == trainkit2.y) / trainkit2.y.shape[0]))
y_pred = predictor.predict(testkit2)
print('Accuracy of test dataset2 is {}'.format(sum(y_pred == testkit2.y) / testkit2.y.shape[0]))




