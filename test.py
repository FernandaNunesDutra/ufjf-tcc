import numpy as np
from sklearn import cluster, datasets, neighbors, metrics, preprocessing, neighbors, metrics

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
    matrix = np.empty((9167,30),dtype=np.int)
    matrix[:] = -1
    for i,row in enumerate(read):
        matrix[row[0]][0] = row[0]
        matrix[row[0]][row[1]] = row[2]
    indexes = []
    for i, row in enumerate(matrix):
        if row[0] == -1:
            indexes.append(i)
    matrix = np.delete(matrix, indexes, axis=0)
    return matrix[:,1:]

def readWatiUserItemMatrix():
    read = np.loadtxt("data_wati_ratings.csv", delimiter=',', skiprows=1, dtype='int')
    matrix = np.empty((1461,10),dtype=np.int)
    matrix[:] = -1
    for i,row in enumerate(read):
        matrix[row[0]][0] = row[0]
        matrix[row[0]][row[1]] = row[2]
    indexes = []
    for i, row in enumerate(matrix):
        if row[0] == -1:
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

def silhouetteScore(data,clustering):
    return metrics.silhouette_score(data,clustering)

def scaleData(data):
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(data)

def similarity(x,y):
    return metrics.pairwise.paired_euclidean_distances(x.reshape(1, -1),y.reshape(1, -1))[0]

def getNeighbors(user, watiUserClusters):

    return users
def main():




    userData, userIndexes = readWatiUserData()
    userItemMatrix = readWatiUserItemMatrix()

    userData = scaleData(userData)
    userClusters = kmeans(userData,10)

    userSimilarityMatrix = np.empty((userData.shape[0],userData.shape[0],))
    for i, u1 in enumerate(userData):
        for j, u2 in enumerate(userData):
            userSimilarityMatrix[i][j] = similarity(u1,u2)

    predictions = np.empty((userItemMatrix.shape[0],userItemMatrix.shape[1],))
    predictions[:] = -1
    for userIndex,userRow in enumerate(predictions):
        cluster = userClusters[userIndex]

        ratingCount = 0
        ratingSum = 0
        for itemIndex, itemRating in enumerate(userRow):
            if userItemMatrix[userIndex][itemIndex] != -1:
                ratingSum += userItemMatrix[userIndex][itemIndex]
                ratingCount += 1
        userMean =  ratingSum/ratingCount

        neighbors = []
        for i,row in enumerate(userClusters):
            if i != userIndex and userClusters[i] == cluster:
                neighbors.append(i)

        for itemIndex, itemRating in enumerate(userRow):
            if userItemMatrix[userIndex][itemIndex] == -1:
                ratingCount= 0
                ratingSum = 0
                for neighborIndex in neighbors:

                    neighborCount = 0
                    neighborSum = 0
                    for itemI, itemR in enumerate(userRow):
                        if userItemMatrix[neighborIndex][itemI] != -1:
                            neighborSum += userItemMatrix[neighborIndex][itemI]
                            neighborCount += 1
                    neighborMean =  neighborSum/neighborCount

                    if userItemMatrix[neighborIndex][itemIndex] != -1:
                        ratingSum += (userItemMatrix[neighborIndex][itemIndex] -  neighborMean)* similarity(userData[userIndex], userData[neighborIndex])
                        neighborCount += 1
                predictions[userIndex][itemIndex] = userMean + ratingSum/neighborCount if neighborCount != 0 else -1

    for userIndex, userRow in enumerate(predictions):
        ratings = userRow.argsort()[-3:][::-1]
        print userIndexes[userIndex],
        for rating in ratings:
            if predictions[userIndex][rating] != -1:
                print rating,
        print \












if __name__ == "__main__":
    main()
