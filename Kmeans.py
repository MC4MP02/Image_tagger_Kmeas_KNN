__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import copy
import numpy as np
from itertools import combinations
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self.WCD = 0
        self.ICD = 0
        self._init_X(X)
        self._init_options(options)  # DICT options
        self.centroids = None
        self.old_centroids = None
        self.labels = None
        self.converged = False

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """

        X = X * 1.0 #Pasamos a float
        if X.ndim == 3:
            self.X = np.reshape(X, (-1, X.shape[-1])) #Si la dimension es 3, transponemos y cambiamos el tamaño
        if X.ndim == 2:
            self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = float(0)
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

    def list_equal(self, a, list):
        #Funcion para comprobar que el array no pertenece a otro array
        for el in list:
            if np.array_equal(el, a):
                return True
        return False

    def _init_centroids(self):
        """
        Initialization of centroids
     """
        elements = [self.X[0]]
        i = 0
        aux = 0
        if self.options['km_init'].lower() == 'first':
            while i < self.K-1:
                if not self.list_equal(self.X[aux], elements):
                    elements.append(self.X[aux])
                    i += 1
                aux += 1
            self.centroids = np.array(elements)
            self.old_centroids = np.array(elements)
        elif self.options['km_init'].lower() == 'random':
            for i in range(self.K):
                self.centroids = np.random.rand(self.K, self.X.shape[1])
                self.old_centroids = np.random.rand(self.K, self.X.shape[1])
        elif self.options['km_init'].lower() == 'custom':
            self.centroids = np.zeros((self.K, self.X.shape[1]))
            self.centroids[0] = self.X[np.random.randint(self.X.shape[0]), :]

            for i in range(1, self.K):
                dist = np.array([min([np.inner(c - x, c - x) for c in self.centroids]) for x in self.X])
                probs = dist / dist.sum()
                cumprobs = probs.cumsum()
                r = np.random.rand()
                ind = np.where(cumprobs >= r)[0][0]
                self.centroids[i] = self.X[ind]

            self.old_centroids = np.copy(self.centroids)

    def get_labels(self):
        """Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        #Calculo de las distancias con los centroides pra despues cojer la classe con distancia minima
        dist = distance(self.X, self.centroids)
        self.labels = np.argmin(dist, axis=1)

    def centroids_calcs(self, a):
        x = np.sum(a[:, 0])
        y = np.sum(a[:, 1])
        z = np.sum(a[:, 2])
        return (x/a.shape[0]), (y/a.shape[0]), (z/a.shape[0])
    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = self.centroids
        self.get_labels()
        list = np.empty([self.K, 3])
        for j in range(self.K):
            list[j] = self.centroids_calcs(np.array(self.X[np.where(self.labels == j)]))
        self.centroids = list

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #Comprobacion si son iguales con una tolerancia(atol) guardada en options('tolerance')
        return np.allclose(self.centroids, self.old_centroids, atol=self.options['tolerance'])

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        self._init_centroids()
        iter = 0
        while True:
            self.get_centroids()
            iter += 1
            if self.converges() is True or iter == self.options['max_iter']:
                break

    def withinClassDistance(self):
        """
        returns the within class distance of the current clustering
        """
        self.WCD = 0
        for i in range(len(self.centroids)):
            self.WCD += np.sum((np.square(self.X[np.where(self.labels == i)] - self.centroids[i]))**2)
        return self.WCD * self.K

    def InterClass(self):
        """
        returns the inter class distance of the current clustering
        """
        comb = list(combinations([i for i in range(self.K)], 2))
        vector = self.centroids
        self.ICD = 0
        for i in range(len(comb)):
            self.ICD += np.sum(np.square(vector[comb[i][0]] - vector[comb[i][1]])**2)
        return self.ICD/self.K

    def find_bestK(self, max_K):
        """
        sets the best k anlysing the results up to 'max_K' clusters
        """
        fst = True
        act = 0
        for i in range(2, max_K):
            if not fst:
                ant = act
            self.K = i
            self.fit()
            interC = self.InterClass()
            intraC = self.withinClassDistance()
            discriminantFisher = intraC / interC
            act = intraC

            if not fst:
                if 100 - 100 * (act / ant) < 20:
                    self.K = i - 1
                    self.fit()
                    break
            fst = False


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    dist = np.linalg.norm(np.expand_dims(X, 2) - np.expand_dims(C.T, 0), axis=1)
    return dist

def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    probs = utils.get_color_prob(centroids) #array con las probabilidades de cada color
    labels = []
    for c in range(len(centroids)):
        max_prob = 0
        i_max_prob = 0
        for i in range(len(probs[c])): #Busqueda de la maxima probabilidad del array
            if probs[c][i] > max_prob:
                max_prob = probs[c][i]
                i_max_prob = i
        labels.append(utils.colors[i_max_prob]) #append del color que corresponde a la maxima probabilidad
    return labels


