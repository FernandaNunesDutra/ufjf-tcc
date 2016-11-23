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

def readAesItemData():
    read = np.loadtxt("data_aes_ratings.csv", delimiter=',', skiprows=1, dtype='int')
    matrix = np.zeros((9167,30),dtype=np.int)

    for i,row in enumerate(read):
        matrix[row[0]][0] = row[0]
        matrix[row[0]][row[1]] = row[2]

    indexes = []
    for i, row in enumerate(matrix):
        if row[0] == 0:
            indexes.append(i)
    matrix = np.delete(matrix, indexes, axis=0)
    matrix = matrix.T
    return matrix[1:,:], [x for x in range(1,30)]

def readWatiItemData():
    read = np.loadtxt("data_wati_ratings.csv", delimiter=',', skiprows=1, dtype='int')
    matrix = np.zeros((1461,10),dtype=np.int)
    for i,row in enumerate(read):
        matrix[row[0]][0] = row[0]
        matrix[row[0]][row[1]] = row[2]
    indexes = []
    for i, row in enumerate(matrix):
        if row[0] == 0:
            indexes.append(i)
    matrix = np.delete(matrix, indexes, axis=0)
    matrix = matrix.T
    return matrix[1:,:], [x for x in range(1,10)]

def readAesUserItemMatrix():
    read = np.loadtxt("data_aes_ratings.csv", delimiter=',', skiprows=1, dtype='int')
    matrix = np.zeros((9167,30),dtype=np.int)
    for i,row in enumerate(read):
        matrix[row[0]][0] = row[0]
        matrix[row[0]][row[1]] = row[2]
    indexes = []
    for i, row in enumerate(matrix):
        if row[0] == 0:
            indexes.append(i)
    matrix = np.delete(matrix, indexes, axis=0)
    return matrix[:,1:]

def readAesUserItemQuitMatrix():
    read = np.loadtxt("data_aes_ratings_quit.csv", delimiter=',', skiprows=1, dtype='int')
    matrix = np.zeros((9167,9),dtype=np.int)
    for i,row in enumerate(read):
        matrix[row[0]][0] = row[0]
        matrix[row[0]][row[1]] = row[2]
    indexes = []
    for i, row in enumerate(matrix):
        if row[0] == 0:
            indexes.append(i)
    matrix = np.delete(matrix, indexes, axis=0)
    return matrix[:,1:]

def readWatiUserItemMatrix():
    read = np.loadtxt("data_wati_ratings.csv", delimiter=',', skiprows=1, dtype='int')
    matrix = np.zeros((1461,10),dtype=np.int)
    for i,row in enumerate(read):
        matrix[row[0]][0] = row[0]
        matrix[row[0]][row[1]] = row[2]
    indexes = []
    for i, row in enumerate(matrix):
        if row[0] == 0:
            indexes.append(i)
    matrix = np.delete(matrix, indexes, axis=0)
    return matrix[:,1:]

def kmeans(data, clusters):
    kmeans = cluster.KMeans(n_clusters=clusters)
    return kmeans.fit_predict(data)

def agglomerative(data, clusters,metric):
    agglomerative = cluster.AgglomerativeClustering(n_clusters=clusters,affinity=metric,linkage='average')
    return agglomerative.fit_predict(data)

def dbscan(data,eps,min_samples):
    dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean',algorithm='brute')
    return dbscan.fit_predict(data)

def calinskiHarabazScore(data, clustering):
    return metrics.calinski_harabaz_score(data,clustering)

def silhouetteScore(data,clustering,metric):
    return metrics.silhouette_score(X=data,labels=clustering,metric=metric)

def scaleData(data):
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(data)


def pearson_affinity(M):
   return 1 - np.array([[pearsonr(a,b)[0] for a in M] for b in M])

def pearson(x,y):
    return pearsonr(x,y)[0]

def similarity(x,y):
    return metrics.pairwise.pairwise_distances(x.reshape(1, -1),y.reshape(1, -1),metric='cosine')[0]

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
    random.seed(datetime.now())

    userData, userIndexes = readAesUserQuitData()
    userItemMatrix = readAesUserItemQuitMatrix()
    print userData.shape
    userData = scaleData(userData)
    nClusters = 2
    userClusters = agglomerative(userData,2,'cosine')
    print userClusters
    print silhouetteScore(userData, userClusters,"cosine")
    print ratingDensity(nClusters, userClusters, userItemMatrix)

    userSimilarityMatrix = np.empty((userData.shape[0],userData.shape[0],))
    print userItemMatrix.shape
    for i, u1 in enumerate(userData):
        for j, u2 in enumerate(userData):
            userSimilarityMatrix[i][j] = similarity(u1,u2)

    predictions = np.zeros((userItemMatrix.shape[0],userItemMatrix.shape[1],))

    for userIndex,userRow in enumerate(predictions):
        if userIndex % 2 == 0:
            neighbors = []
            for i,cluster in enumerate(userClusters):
                if i != userIndex and cluster == userClusters[userIndex]:
                    neighbors.append(i)

            for itemIndex, itemRating in enumerate(userRow):
                if userItemMatrix[userIndex][itemIndex] == 0:
                    ratingSum = 0
                    for neighborIndex in neighbors:
                        sim = similarity(userData[userIndex],userData[neighborIndex])
                        #if(sim == 0):
                        #    ratingSum += userItemMatrix[neighborIndex][itemIndex]
                        #else:
                        ratingSum += userItemMatrix[neighborIndex][itemIndex] * (sim)
                    predictions[userIndex][itemIndex] = ratingSum
        else:
            randItens = [x for x in range(userItemMatrix.shape[1])]
            random.shuffle(randItens)
            count = 0
            for item in randItens:
                if userItemMatrix[userIndex][item] == 0:
                    predictions[userIndex][item] = 1
                    count += 1
                if(count == 3):
                    break


    for userIndex, userRow in enumerate(predictions):
        recommendations = userRow.argsort()[-3:][::-1]
        print userIndexes[userIndex],
        for recommendation in recommendations:
            if predictions[userIndex][recommendation] != -1:
                print recommendation + 1,
        print \

    print "RANDOMIZED"
    for userIndex, userRow in enumerate(predictions):
        if(userIndex % 2 == 1):
            recommendations = userRow.argsort()[-3:][::-1]
            print userIndexes[userIndex],
            for recommendation in recommendations:
                if predictions[userIndex][recommendation] != -1:
                    print recommendation + 1,
            print \

    print "RECOMMENDED"
    for userIndex, userRow in enumerate(predictions):
        if(userIndex % 2 == 0):
            recommendations = userRow.argsort()[-3:][::-1]
            print userIndexes[userIndex],
            for recommendation in recommendations:
                if predictions[userIndex][recommendation] != -1:
                    print recommendation + 1,
            print \



if __name__ == "__main__":
    main()