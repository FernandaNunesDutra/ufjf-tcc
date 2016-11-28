import numpy as np
import random
from sklearn import cluster, datasets, neighbors, metrics, preprocessing, neighbors, metrics
from scipy.stats import pearsonr
from datetime import datetime

def readUserData(dataset):
    if (dataset == 'aes'):
        file_name = "aes_users.csv"
    if (dataset == 'wati'):
        file_name = "wati_users.csv"
    read = np.loadtxt(file_name, delimiter=',', skiprows=1)
    temp = read.view(np.ndarray)
    np.lexsort((temp[:, 0], ))
    read = temp[np.lexsort((temp[:, 0], ))]
    return read[:, 1:], read[:,0].astype(int)

def readUserItemMatrix(dataset):
    if (dataset == 'aes'):
        file_name = "aes_ratings.csv"
        m = 9167
        n = 9
    if (dataset == 'wati'):
        file_name = "wati_ratings.csv"
        m = 1461
        n = 10
    read = np.loadtxt(file_name, delimiter=',', skiprows=1, dtype='int')
    matrix = np.zeros((m,n),dtype=np.int)
    for i,row in enumerate(read):
        matrix[row[0]][0] = row[0]
        matrix[row[0]][row[1]] = row[2]
    indexes = []
    for i, row in enumerate(matrix):
        if row[0] == 0:
            indexes.append(i)
    matrix = np.delete(matrix, indexes, axis=0)
    return matrix[:,1:]

def computeClusters(data,algorithm,clusters,metric):
    if(algorithm == 'kmeans'):
        kmeans = cluster.KMeans(n_clusters=clusters)
        return kmeans.fit_predict(data)
    if(algorithm == 'agglomerative'):
        agglomerative = cluster.AgglomerativeClustering(n_clusters=clusters,affinity=metric,linkage='average')
        return agglomerative.fit_predict(data)


def silhouetteScore(data,clustering,metric):
    return metrics.silhouette_score(X=data,labels=clustering,metric=metric)

def scaleData(data):
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(data)

def similarity(x,y,metric):
    if(metric == 'manhattan' or metric == 'euclidean'):
        dist = metrics.pairwise.pairwise_distances(x.reshape(1, -1),y.reshape(1, -1),metric=metric)[0]
        if dist == 0:
            return 1;
        else:
            return 1/dist
    else:
        return metrics.pairwise.pairwise_distances(x.reshape(1, -1),y.reshape(1, -1),metric=metric)[0]

def raters(nClusters, userClusters, userItemMatrix):
    ratingRatio = []
    for cluster in range(nClusters):
        raterCount = 0
        neighborCount = 0
        for userIndex, ratings in enumerate(userItemMatrix):
            if userClusters[userIndex] == cluster:
                neighborCount += 1
                if np.sum(ratings) != 0:
                    raterCount += 1
        ratingRatio.append((raterCount,neighborCount))
    return ratingRatio

def clusterAnalysis(dataset):
    userData, userIndexes = readUserData(dataset)
    userData = scaleData(userData)

    print 'KMEANS'
    for nClusters in range(2,11):
        userClusters = computeClusters(userData,'kmeans',nClusters,'euclidean')
        print silhouetteScore(userData, userClusters,"euclidean")
    print '============================'

    print 'AGGLOMERATIVE EUCLIDEAN'
    for nClusters in range(2,11):
        userClusters = computeClusters(userData,'agglomerative',nClusters,'euclidean')
        print silhouetteScore(userData, userClusters,"euclidean")
    print '============================'

    print 'AGGLOMERATIVE COSINE'
    for nClusters in range(2,11):
        userClusters = computeClusters(userData,'agglomerative',nClusters,'cosine')
        print silhouetteScore(userData, userClusters,"cosine")
    print '============================'

    print 'AGGLOMERATIVE MANHATTAN'
    for nClusters in range(2,11):
        userClusters = computeClusters(userData,'agglomerative',nClusters,'manhattan')
        print silhouetteScore(userData, userClusters,"manhattan")
    print '============================'

def computeRecommendations(dataset,algorithm,nClusters,metric):

    userData, userIds = readUserData(dataset)
    userItemMatrix = readUserItemMatrix(dataset)

    userData = scaleData(userData)

    userClusters = computeClusters(userData,algorithm,nClusters,metric)
    print userClusters
    print silhouetteScore(userData, userClusters,metric)
    print raters(nClusters, userClusters, userItemMatrix)

    userSimilarityMatrix = np.empty((userData.shape[0],userData.shape[0],))
    for i, u1 in enumerate(userData):
        for j, u2 in enumerate(userData):
            userSimilarityMatrix[i][j] = similarity(u1,u2,metric)

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
                        sim = similarity(userData[userIndex],userData[neighborIndex],metric)
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
        print userIds[userIndex],
        for recommendation in recommendations:
            if predictions[userIndex][recommendation] > 0:
                print recommendation + 1 ,
            else:
                print 0 ,
        print \

def main():
    random.seed(datetime.now())

    dataset = 'aes'
    nclusters = 2
    metric = 'cosine'
    algorithm = 'agglomerative'

    computeRecommendations(dataset,algorithm,nclusters,metric)
    #clusterAnalysis('wati')


if __name__ == "__main__":
    main()
