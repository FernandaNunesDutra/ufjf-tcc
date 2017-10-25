import numpy as np
import random
from sklearn import cluster, datasets, neighbors, metrics, preprocessing, neighbors, metrics
from scipy.stats import pearsonr
from datetime import datetime


def readAesUserQuitData():
    read = np.loadtxt("data_aes_users_quit.csv", delimiter=',', skiprows=1)
    return read[:, 1:], read[:,0].astype(int)

def readAesUserCutData():
    read = np.loadtxt("data_aes_users_cut.csv", delimiter=',', skiprows=1)
    return read[:, 1:], read[:,0].astype(int)

def readAesUserData():
    read = np.loadtxt("data_aes_users.csv", delimiter=',', skiprows=1)
    return read[:, 1:], read[:,0].astype(int)


def readWatiUserData():
    read = np.loadtxt("data_wati_users.csv", delimiter=',', skiprows=1)
    return read[:, 1:], read[:,0].astype(int)

def kmeans(data, clusters):
    kmeans = cluster.KMeans(n_clusters=clusters)
    return kmeans.fit_predict(data)

def agglomerative(data, clusters,metric):
    agglomerative = cluster.AgglomerativeClustering(n_clusters=clusters,affinity=metric,linkage='average')
    return agglomerative.fit_predict(data)

def silhouetteScore(data,clustering,metric):
    return metrics.silhouette_score(X=data,labels=clustering,metric=metric)

def scaleData(data):
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(data)

def ratingDensity(nClusters, userClusters, userItemMatrix):
    ratingRatio = []
    for cluster in range(nClusters):
        raterCount = 0
        neighborCount = 0
        for userIndex, ratings in enumerate(userItemMatrix):
            if userClusters[userIndex] == cluster:
                neighborCount += 1
                if np.sum(ratings) != 0:
                    raterCount += 1
                #for rating in ratings:
                #    if rating != 0:
                #        ratingCount += 1
                #if ratingCount != 0:
                #    neighborCount += 1
        ratingRatio.append((raterCount,neighborCount))
    return ratingRatio

def main():

    userData, userIndexes = readAesUserQuitData()
    userData = scaleData(userData)

    print 'KMEANS'
    for nClusters in range(2,11):
        userClusters = kmeans(userData,nClusters)
        #print ratingDensity(nClusters, userClusters, userItemMatrix)
        print silhouetteScore(userData, userClusters,"euclidean")
    print '============================'

    print 'AGGLOMERATIVE EUCLIDEAN'
    for nClusters in range(2,11):
        userClusters = agglomerative(userData,nClusters,"euclidean")
        #print ratingDensity(nClusters, userClusters, userItemMatrix)
        print silhouetteScore(userData, userClusters,"euclidean")
    print '============================'

    print 'AGGLOMERATIVE COSINE'
    for nClusters in range(2,11):
        userClusters = agglomerative(userData,nClusters,"cosine")
        #print ratingDensity(nClusters, userClusters, userItemMatrix)
        print silhouetteScore(userData, userClusters,"cosine")
    print '============================'

    print 'AGGLOMERATIVE MANHATTAN'
    for nClusters in range(2,11):
        userClusters = agglomerative(userData,nClusters,"manhattan")
        #print ratingDensity(nClusters, userClusters, userItemMatrix)
        print silhouetteScore(userData, userClusters,"manhattan")
    print '============================'

if __name__ == "__main__":
    main()
