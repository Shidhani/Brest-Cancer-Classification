#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:32:53 2020


"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import operator
import mglearn
import time
import seaborn as sns

"""
Loading the dataset 
"""
cancer = datasets.load_breast_cancer()

print("Features Names:")
print(cancer.feature_names)
print()
print("Target Names:")
print(cancer.target_names)

#df 
#df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
label=cancer["target"]
df.head()
x = cancer['data']
y = cancer["target"]
"""
Split dataset into training and test subsets.
80 percent will be used for training, 20 percent for testing
"""


# 
#x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
#
#print(x_train, y_train)

x_train, x_test, y_train, y_test = [],[],[],[]
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)  



def split(x):
    from sklearn.model_selection import train_test_split
    global x_train
    global x_test
    global y_train
    global y_test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



"""
Apply K-NN classifier for k = 1, 3 and 5 
"""

def getEuclidDist2Vectors(v1, v2):
    return np.sqrt(np.sum((v1-v2)**2))

def KNN(x_train, y_train, x_test, K):
 
    y_predicted = []
    distances = {}
    sz_train = x_train.shape[0]   # number of test samples in x_train
    
    # for each test sample.....
    for xtst in x_test:
        #... compute the distance between xtst it to each train sample
        for i in range(sz_train):
            dst = getEuclidDist2Vectors(xtst, x_train[i,:])
            distances[i] = dst
        
        # sort the distances from smalest to largest
        distSorted = sorted(distances.items(), key=operator.itemgetter(1))
     
        # get the k neares neighbors in a list <neighbors>
        neighbors = []
        for k in range(K):
            neighbors.append(distSorted[k][0])
            
        # get the labels of the first nearest neighbors from the x_train in list <lbl>
        lbl = []
        votes = {}
        
        for n in neighbors:
            l = y_train[n]
            lbl.append(y_train[n])
            if l in votes:
                votes[l] +=1
            else:
                votes[l] = 1
     
        votesSorted = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
        dominantLabel = votesSorted[0][0]
        
        y_predicted.append(dominantLabel)
        
    return y_predicted
        
y_predict_k1 = KNN(x_train, y_train, x_test, 1)
y_predict_k3 = KNN(x_train, y_train, x_test, 3)
y_predict_k5 = KNN(x_train, y_train, x_test, 5)

def get_feature_accuracy(X,k):
    split(X)
    y_pred = KNN(x_train, y_train, x_test, k)
    acc_score = accuracy_score(y_test, y_pred)
    return acc_score

#mglearn.plots.plot_knn_classification(n_neighbors=1)

k1 = get_feature_accuracy(x,1)

k3 = get_feature_accuracy(x,3)

k5 = get_feature_accuracy(x,5)
#	Plot the relationship between k and acc. 

title = "Plot the relationship between k and acc"
print(title.center(100, '='))

print("Accuracy for k = 1: ", k1)
print("Accuracy for k = 3: ", k3)
print("Accuracy for k = 5: ", k5)

plt.figure(figsize=(4, 4))
plt.plot([1,3,5], [k1,k3,k5])
plt.suptitle('k value and acc')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.show()


"""
Select the best features for each k using GA 
For each k report the best feature selected. 
"""


#import the breast cancer dataset 



#splitting the model into training and testing set
x_train, x_test, y_train, y_test = train_test_split(df, 
                                                    label, test_size=0.20, 
                                                    random_state=101)
logmodel = 0

#training KNN model for K =1
def train(k):
    global logmodel
    logmodel = KNeighborsClassifier(n_neighbors=k)
    logmodel.fit(x_train,y_train)
    predictions = logmodel.predict(x_test)
    print("Accuracy score before GA is = "+ str(accuracy_score(y_test,predictions)))




#defining various steps required for the genetic algorithm
def initilization_of_population(size,n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat,dtype=np.bool)
        chromosome[:int(0.3*n_feat)]=False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population

def fitness_score(population):
    scores = []
    for chromosome in population:
        logmodel.fit(x_train.iloc[:,chromosome],y_train)
        predictions = logmodel.predict(x_test.iloc[:,chromosome])
        scores.append(accuracy_score(y_test,predictions))
    scores, population = np.array(scores), np.array(population) 
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds,:][::-1])

def selection(pop_after_fit,n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen

def crossover(pop_after_sel):
    population_nextgen=pop_after_sel
    for i in range(len(pop_after_sel)):
        child=pop_after_sel[i]
        child[3:7]=pop_after_sel[(i+1)%len(pop_after_sel)][3:7]
        population_nextgen.append(child)
    return population_nextgen

def mutation(pop_after_cross,mutation_rate):
    population_nextgen = []
    for i in range(int(len(pop_after_cross)/2),len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        if random.random() < mutation_rate:
            mask = np.random.rand(len(chromosome)) < 0.05
            chromosome[mask] = False
        population_nextgen.append(chromosome)
    return population_nextgen

def generations(size,n_feat,n_parents,mutation_rate,n_gen,X_train,
                                   X_test, y_train, y_test):
    best_chromo= []
    best_score= []
    population_nextgen=initilization_of_population(size,n_feat)
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen)
        print(scores[:2])
        pop_after_sel = selection(pop_after_fit,n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross,mutation_rate)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo,best_score


def GA(k):
    train(k)
    chromo,score=generations(size=200,n_feat=30,n_parents=100,mutation_rate=0.10,
                         n_gen=38,X_train=x_train,X_test=x_test,y_train=y_train,y_test=y_test)
    logmodel.fit(x_train.iloc[:,chromo[-1]],y_train)
    predictions = logmodel.predict(x_test.iloc[:,chromo[-1]])
    print("Accuracy score after GA is= "+str(accuracy_score(y_test,predictions)))
    selected_features = list(x_test.iloc[:,chromo[-1]])
    print()
    print("Feature Selected:")
    i=1
    for feature in selected_features:
        print(str(i)+". "+feature)
        i+=1
#    print(selected_features)
    
title = " GA for k = 1 "
print(title.center(100, '='))
GA(1)

title = " GA for k = 3 "
print(title.center(100, '='))
GA(3)

title = " GA for k = 5 "
print(title.center(100, '='))
GA(5)

#print(predictions)

#selected = []
#values= x_test.values.tolist()
#for i in range(len(predictions)):
#    if predictions[i] == 1:
#        selected.append(values[i])

#f= max(score)
#print(f)
#print(chromo[score.index(f)])
    