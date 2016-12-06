import numpy as np
import random
from sys import argv
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

def scaleData(data):
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(data)

def computePredictions(userIds,userIdsTest,userData,userDataTest,userItemMatrix):
    predictions = np.zeros((userDataTest.shape[0],userItemMatrix.shape[1],))

    for userIndex,userRow in enumerate(predictions):

            for itemIndex, itemRating in enumerate(userRow):
                ratingSum = 0
                rand =  random.uniform(-1, 1)
                predictions[userIndex][itemIndex] = rand
    return predictions

def computeRecommendations(predictions, userItemMatrixTest):
    recommendations = np.zeros((predictions.shape[0],predictions.shape[1],),dtype='int')
    recommendations[:] = -1
    for userIndex, ratings in enumerate(predictions):
        #top3 = ratings.argsort()[-3:][::-1]
        sortedRatings = np.sort(ratings)
        count = 0
        for i, rating in enumerate(sortedRatings):
            if userItemMatrixTest[userIndex][i] != 0:
                recommendations[userIndex][i] = 1
                count +=1
            if count == 3:
                break
    return recommendations


def main():


    dataset = argv[1]
    seed = int(argv[2])

    random.seed(seed)


    userData, userIds = readUserData(dataset)
    userData = scaleData(userData)
    userItemMatrix, userIds2 = readUserItemMatrix(dataset)

    labels=[]
    for user in userItemMatrix:
        if -1 in user:
            labels.append(-1)
        else:
            labels.append(1)

    averagePrecision = []
    averageRecall = []
    averageFmeasure = []
    averagePrecision1 = []
    averageRecall1 = []
    averageFmeasure1 = []

    kf = model_selection.StratifiedKFold(n_splits = 5, shuffle=True)
    for trainIndex, testIndex in kf.split(userData,labels):
        userDataTrain, userDataTest = userData[trainIndex], userData[testIndex]
        userIdsTrain, userIdsTest = userIds[trainIndex], userIds[testIndex]
        userItemMatrixTrain, userItemMatrixTest = userItemMatrix[trainIndex], userItemMatrix[testIndex]

        predictions = computePredictions(userIds,userIdsTest,userData,userDataTest,userItemMatrix)
        recommendations = computeRecommendations(predictions,userItemMatrixTest)


        precision = []
        recall = []
        fmeasure =[ ]
        for i, ratings in enumerate(userItemMatrixTest):
            truth = []
            predicted = []
            for j, rating in enumerate(ratings):
                if rating != 0:
                    truth.append(rating)
                    predicted.append(recommendations[i][j])
            precision.append(metrics.precision_score(truth, predicted, labels=[1,-1], pos_label=1, average='binary'))
            recall.append(metrics.recall_score(truth, predicted, labels=[1,-1], pos_label=1, average='binary'))
            fmeasure.append(metrics.f1_score(truth, predicted, labels=[1,-1], pos_label=1, average='binary'))
        averagePrecision.append(sum(precision)/len(precision))
        averageRecall.append(sum(recall)/len(recall))
        averageFmeasure.append(sum(fmeasure)/len(fmeasure))



        precision = []
        recall = []
        fmeasure =[ ]
        for i, ratings in enumerate(userItemMatrixTest):
            truth = []
            predicted = []
            for j, rating in enumerate(ratings):
                if rating != 0:
                    truth.append(rating)
                    predicted.append(1) if predictions[i][j] > 0 else predicted.append(-1)
            precision.append(metrics.precision_score(truth, predicted, labels=[1,-1], pos_label=1, average='binary'))
            recall.append(metrics.recall_score(truth, predicted, labels=[1,-1], pos_label=1, average='binary'))
            fmeasure.append(metrics.f1_score(truth, predicted, labels=[1,-1], pos_label=1, average='binary'))
        averagePrecision1.append(sum(precision)/len(precision))
        averageRecall1.append(sum(recall)/len(recall))
        averageFmeasure1.append(sum(fmeasure)/len(fmeasure))


    print sum(averagePrecision)/len(averagePrecision)
    print sum(averageRecall)/len(averageRecall)
    print sum(averageFmeasure)/len(averageFmeasure)
    print sum(averagePrecision1)/len(averagePrecision1)
    print sum(averageRecall1)/len(averageRecall1)
    print sum(averageFmeasure1)/len(averageFmeasure1)



if __name__ == "__main__":
    main()
