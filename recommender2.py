import numpy as np
import random
from sklearn import cluster, datasets, neighbors, metrics, preprocessing, neighbors, metrics, model_selection
from scipy.stats import pearsonr
from datetime import datetime

def readUserData(dataset):
    if (dataset == 'aes'):
        file_name = "aes_users_raters.csv"
    if (dataset == 'wati'):
        file_name = "wati_users_raters.csv"
    read = np.loadtxt(file_name, delimiter=',', skiprows=1)
    temp = read.view(np.ndarray)
    np.lexsort((temp[:, 0], ))
    read = temp[np.lexsort((temp[:, 0], ))]
    return read[:, 1:], read[:,0].astype(int)

def readUserItemMatrix(dataset):
    if (dataset == 'aes'):
        file_name = "aes_ratings_raters.csv"
        m = 9167
        n = 9
    if (dataset == 'wati'):
        file_name = "wati_ratings_raters.csv"
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
    return matrix[:,1:], read[:,0]

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

def computePredictions(userIds,userIdsTest,userData,userDataTest,userItemMatrix,userClusters,userClustersTest,metric):
    predictions = np.zeros((userDataTest.shape[0],userItemMatrix.shape[1],))

    for userIndex,userRow in enumerate(predictions):

            neighbors = []
            for i,cluster in enumerate(userClusters):
                if userIds[i] != userIdsTest[userIndex] and cluster == userClustersTest[userIndex]:
                    neighbors.append(i)

            for itemIndex, itemRating in enumerate(userRow):
                ratingSum = 0
                simSum = 0
                for neighborIndex in neighbors:
                    sim = similarity(userDataTest[userIndex],userData[neighborIndex],metric)
                    if userItemMatrix[neighborIndex][itemIndex] != 0:
                        simSum += sim
                    ratingSum += userItemMatrix[neighborIndex][itemIndex] * (sim)
                if simSum == 0:
                    simSum = 1
                predictions[userIndex][itemIndex] = ratingSum/simSum

    return predictions

def computeRecommendations(predictions):
    recommendations = np.zeros((predictions.shape[0],3,))
    for userIndex, userRow in enumerate(predictions):
        top3 = userRow.argsort()[-3:][::-1]
        recommendations[userIndex] = top3
    return recommendations

def main():
    random.seed(datetime.now())

    dataset = 'aes'
    nClusters = 2
    metric = 'cosine'
    algorithm = 'agglomerative'

    print dataset, nClusters, metric, algorithm

    userData, userIds = readUserData(dataset)
    userData = scaleData(userData)
    userItemMatrix, userIds2 = readUserItemMatrix(dataset)
    userClusters = computeClusters(userData,algorithm,nClusters,metric)

    labels=[]
    for user in userItemMatrix:
        if -1 in user:
            labels.append(-1)
        else:
            labels.append(1)


    print 'SILHOUETTE SCORE'
    print silhouetteScore(userData,userClusters,metric)

    kf = model_selection.StratifiedKFold(n_splits = 5, shuffle=False)
    #matrix = np.zeros((2,2,),dtype=int)
    recall = []
    precision = []
    for trainIndex, testIndex in kf.split(userData, labels):

        userDataTrain, userDataTest = userData[trainIndex], userData[testIndex]
        userIdsTrain, userIdsTest = userIds[trainIndex], userIds[testIndex]
        userItemMatrixTrain, userItemMatrixTest = userItemMatrix[trainIndex], userItemMatrix[testIndex]
        userClustersTrain, userClustersTest = userClusters[trainIndex], userClusters[testIndex]

        predictions = computePredictions(userIds,userIdsTest,userData,userDataTest,userItemMatrix,userClusters,userClustersTest,metric)

        predicted = []
        truth = []
        for i, ratings in enumerate(userItemMatrixTest):
            for j, rating in enumerate(ratings):
                if rating != 0:
                    truth.append(rating)
                    predicted.append(1) if predictions[i][j] > 0 else predicted.append(-1)

        #matrix_aux =  metrics.confusion_matrix(truth, predicted,labels=[1,-1])
        recall.append(metrics.recall_score(truth, predicted, labels=[1,-1], pos_label=1, average='binary'))
        precision.append(metrics.precision_score(truth, predicted, labels=[1,-1], pos_label=1, average='binary'))

        #for i,row in enumerate(matrix):
        #    for j,elem in enumerate(row):
        #        matrix[i][j] += matrix_aux[i][j]
    print 'AVERAGE RECALL'
    print sum(recall)/len(recall)
    print 'AVERAGE PRECISION'
    print sum(recall)/len(recall)



if __name__ == "__main__":
    main()
