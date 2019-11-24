import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import time
class latent_factor():

    def __init__(self, K, alpha, beta, iterations):
        """
        Initialize the arguements provided to the class.
        The arguements being -
        K = Number of factors
        alpha = learning rate
        beta = regularization parameter
        iterations = Number of iterations
        """
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        
    def get_dataset_directory(self):
        curr_dir = os.getcwd()  # gets the current working directory
        dataset_dir = os.path.join(curr_dir, "dataset")  # concatenates
        return dataset_dir
    
    def initialise(self):
        """
        Prepares the dataset for further operations.
        """ 
        
        dataset_directory = self.get_dataset_directory()
        dataset = os.path.join(dataset_directory, "test_ratings.dat")
        file = open(dataset, "r")
        data = [] 
        for line in file:
            fields = line.split("::")
            userid = int(fields[0])-1
            movieid = int(fields[1])-1
            rating = int(fields[2])
            data.append([userid, movieid, rating])
        df = pd.DataFrame(data, columns=['userid', 'movieid', 'rating'])
        self.num_users = max(df['userid'])+1
        self.num_items = max(df['movieid'])+1
        self.samples = np.array((df['userid'], df['movieid'], df['rating']))
        self.samples = np.transpose(self.samples)
        
    def build_initial_matrix(self):
        """
        Build an initial rating matrix based on the data points.
        """
        self.R = np.zeros((self.num_users, self.num_items))
        for user, movie, rating in self.samples:
            self.R[user][movie] = rating
        print("Initial Rating Matrix:")
        print(self.R)
        
        
    def calc_b(self):
        """ 
        Calculate the global mean
        """ 
        
        bias_global = 0
        cnt = 0
        for user, movie, rating in self.samples:
            if rating != 0:
                bias_global += rating
                cnt += 1
        bias_global = bias_global/cnt
        self.b = bias_global

    def train(self):
        """
        main funtion used for training dataset and acts as a driver function for functions named 'sgd' and 'mae_rmse'
        """
        # Initialize user matrix P and item matrix Q with zerps
        self.P = np.random.normal(size=(self.num_users, self.K), scale = 1/self.K)
        self.Q = np.random.normal(size=(self.num_items, self.K), scale = 1/self.K)
        

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        
        #to calculate global avergae bias
        self.calc_b()
        

        # Perform stochastic gradient descent for number of iterations
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
#             print(self.samples)
            self.sgd()
        
        print("Final Rating Prediction Matrix:")
        self.final_rating_matrix = self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)
        print(self.final_rating_matrix)
        RMSE, MAE = self.mae_rmse()
        print("RMSE: ", RMSE)
        print("MAE: ", MAE)

    def mae_rmse(self):
        """
        A function to compute the mean average error and total mean square error
        """
        xs, ys = self.R.nonzero()
        rmse_error = 0
        mean_error = 0
        for x, y in zip(xs, ys):
            rmse_error += pow(self.R[x, y] - self.final_rating_matrix[x, y], 2)
            mean_error += abs(self.R[x, y] - self.final_rating_matrix[x, y])
        n = len(self.samples)
        return np.sqrt(rmse_error/n), mean_error/n
    

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for user, movie, rating in self.samples:
            # Computer prediction and error
            
            #predicted rating for user i and item j
            prediction = self.b + self.b_u[user] + self.b_i[movie] + self.P[user, :].dot(self.Q[movie, :].T)
            
            #error between actual rating and predicted rating
            error = (rating - prediction)

            # Update biases
            self.b_u[user] += self.alpha * (error - self.beta * self.b_u[user])
            self.b_i[movie] += self.alpha * (error - self.beta * self.b_i[movie])

            # Update user and item latent feature matrices
            self.P[user, :] += self.alpha * (error * self.Q[movie, :] - self.beta * self.P[user,:])
            self.Q[movie, :] += self.alpha * (error * self.P[user, :] - self.beta * self.Q[movie,:])
            
start = time.time()
model = latent_factor(K=2, alpha=0.1, beta=0.01, iterations=50)
model.initialise()
model.build_initial_matrix()
model.train()
end = time.time() - start
print("TIME: ", end)