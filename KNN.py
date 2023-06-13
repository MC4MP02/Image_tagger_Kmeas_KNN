__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist

class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        self.neighbors = []

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        train_data = train_data.astype(float) #conversion a float
        n = train_data.shape[0]
        train_data = train_data.reshape(n, -1) #reshape de la matriz de entreno
        self.train_data = train_data #actualizacion de la nueva matriz

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        test_data = test_data.astype(float) #conversion a float
        N = test_data.shape[0]
        K = test_data.shape[1]
        P = test_data.shape[2]
        E = test_data.shape[3]
        dim = K*P*E
        #dim = test_data.shape[1]*test_data.shape[2] #dim = N*P
        #test_data = test_data.reshape((test_data.shape[0], dim)) #reshape de la matriz a un array de N*dim
        r_test_data = test_data.reshape((N, dim))
        distances = cdist(r_test_data, self.train_data, 'cityblock') #calculo de las distancias con cdist del test_data con el array de train (Manhattan)
        indices = np.argsort(distances, axis=1)[:, :k]
        self.neighbors = self.labels[indices] #actualizacion de los nuevos neihbors

    def get_class(self):
        """
            Get the class by maximum voting
            : return : numpy array of Nx1 elements .
                For each of the rows in self . neighbors gets the most
                voted value ( i . e . the class at which that row belongs )
        """
        labels = {}
        class_list = []
        for row in self.neighbors:
            for i in range(len(row)):
                if row[i] not in labels:
                    labels[row[i]] = 1 #init row[i]
                else:
                    labels[row[i]] += 1 #row[i]++
            class_list.append(max(labels, key=labels.get)) #el .get devuelve la key asociada al value maximo encontrado por max()
            labels = {}
        return np.array(class_list) #return un numpy array la lista con los maximos keys
    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the class 2nd the  % of votes it got
        """
        self.get_k_neighbours(test_data, k) #init neighbours
        return self.get_class() #make classes
