#importing required modules
import pandas as pd
import numpy as np

#path variables
path1_train = "/home/megh-ops/Documents/data/ds1_train.csv"
path1_test = "/home/megh-ops/Documents/data/ds1_test.csv"
path2_train = "/home/megh-ops/Documents/data/ds2_train.csv"
path2_test = "/home/megh-ops/Documents/data/ds2_test.csv"

#functions to calculate the parameters of the model and to predict the output
#the member variables are sigma, phi, mu0 ,mu1, theta_0 and theta
class Predictor:

    def __init__(self, threshold):
        self.threshold = threshold

    #sigmoid function
    def sigmoid(self, testkit):
        z = np.dot(self.theta.T, testkit.x.T) + self.theta_0
        return np.reshape(1/ (1 + np.exp(-z)), (testkit.x.shape[0], 1))

    #methods to compute all the parameters of the model
    def phi_calc(self, trainingkit):
        sum = 0
        for i in range(trainingkit.y.shape[0]):
            if(trainingkit.y[i] == 1):
                sum += 1
        self.phi = sum / (trainingkit.y.shape[0])

    def mu_calc(self, trainingkit):
        count1 = 0
        count2 = 0
        mu_0 = np.zeros(trainingkit.x.shape[1], dtype = float)
        mu_1 = np.zeros(trainingkit.x.shape[1], dtype = float)
        for i in range(trainingkit.y.shape[0]):
            if(trainingkit.y[i] == 0):
                mu_0 += trainingkit.x[i]
                count1 += 1
            else:
                mu_1 += trainingkit.x[i]
                count2 += 1
        self.mu0 = np.reshape((mu_0 / count1), (trainingkit.x.shape[1], 1))
        self.mu1 = np.reshape((mu_1 / count2), (trainingkit.x.shape[1], 1))

    def sigma_calc(self, trainingkit):
        sigma = np.zeros((trainingkit.x.shape[1], trainingkit.x.shape[1]), dtype = float)
        for i in range(trainingkit.y.shape[0]):
            xi = np.reshape(trainingkit.x[i,:], (trainingkit.x.shape[1], 1))
            if(trainingkit.y[i] == 1):
                z = xi - self.mu1
                sigma += np.dot(z, z.T)
            else:
                w = xi - self.mu0
                sigma += np.dot(w, w.T)
        self.sigma = sigma / (trainingkit.y.shape[0])

    #theta and theta_0 are calculated using the expressions obtained in subtask3 
    def theta_0_calc(self):
        self.theta_0 = np.log(self.phi/(1 - self.phi)) + (1/2) * (np.dot(self.mu0.T, np.dot(np.linalg.inv(self.sigma), self.mu0)) - np.dot(self.mu1.T, np.dot(np.linalg.inv(self.sigma), self.mu1)))

    def theta_calc(self):
        self.theta = np.dot(np.linalg.inv(self.sigma), (self.mu1 - self.mu0))

    #method that calls other methods to compute all the parameters
    def learn(self, trainingkit):
        self.phi_calc(trainingkit)
        self.mu_calc(trainingkit)
        self.sigma_calc(trainingkit)
        self.theta_0_calc()
        self.theta_calc()

    #this takes in the features of the testdata and predicts the output
    def predict(self, testkit):
        result = self.sigmoid(testkit)
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
#has member variables x(features) and y(target)
class TrainingKit:
    def __init__(self, path):
        dframe = pd.read_csv(path)
        dframe_x = dframe[['x_1', 'x_2']]
        self.x = np.array(dframe_x, dtype = float)
        dframe_y = dframe[['y']]
        self.y = np.array(dframe_y, dtype = float)

#a child class of trainingkit
#only used for predicting the target and not to calculate the parameters of the model
class Testkit(TrainingKit):
    pass

#setting the threshold to classify
threshold = 0.5
predictor = Predictor(threshold)

#creating instances of training and test data
trainkit1 = TrainingKit(path1_train)
testkit1 = Testkit(path1_test)

trainkit2 = TrainingKit(path2_train)
testkit2 = Testkit(path2_test)

#learn method calculates the parameters of the model using training data
predictor.learn(trainkit1)
#predict method returns an array of expected values which is then compared to the expected output to compute the accuracy
y_pred = predictor.predict(trainkit1)
print('Accuracy of training dataset1 is {}'.format(sum(y_pred == trainkit1.y) / trainkit1.y.shape[0]))
y_pred = predictor.predict(testkit1)
print('Accuracy of test dataset1 is {}'.format(sum(y_pred == testkit1.y) / testkit1.y.shape[0]))

predictor.learn(trainkit2)
y_pred = predictor.predict(trainkit2)
print('Accuracy of training dataset2 is {}'.format(sum(y_pred == trainkit2.y) / trainkit2.y.shape[0]))
y_pred = predictor.predict(testkit2)
print('Accuracy of test dataset2 is {}'.format(sum(y_pred == testkit2.y) / testkit2.y.shape[0]))














