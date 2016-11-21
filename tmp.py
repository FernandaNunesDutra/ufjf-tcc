userSimilarityMatrix = np.empty((userData.shape[0],userData.shape[0],))
for i, u1 in enumerate(userData):
    for j, u2 in enumerate(userData):
        userSimilarityMatrix[i][j] = similarity(u1,u2)
print userSimilarityMatrix

predictions = np.zeros((userItemMatrix.shape[0],userItemMatrix.shape[1],))
for userIndex,userRow in enumerate(predictions):

    #ratingCount = 0
    #ratingSum = 0
    #for itemIndex, itemRating in enumerate(userRow):
    #    if userItemMatrix[userIndex][itemIndex] != 0:
    #        ratingSum += userItemMatrix[userIndex][itemIndex]
    #        ratingCount += 1
    #userMean =  ratingSum/float(ratingCount)

    neighbors = []
    for i,cluster in enumerate(userClusters):
        if i != userIndex and cluster == userClusters[userIndex]:
            neighbors.append(i)

    for itemIndex, itemRating in enumerate(userRow):
        if userItemMatrix[userIndex][itemIndex] == 0:
            ratingCount= 0
            ratingSum = 0
            for neighborIndex in neighbors:

                #neighborCount = 0
                #neighborSum = 0
                #for itemI, itemR in enumerate(userRow):
                #    if userItemMatrix[neighborIndex][itemI] != 0:
                #        neighborSum += userItemMatrix[neighborIndex][itemI]
                #        neighborCount += 1
                #neighborMean =  neighborSum/float(neighborCount)

                #if userItemMatrix[neighborIndex][itemIndex] != 0:
                ratingSum += (userItemMatrix[neighborIndex][itemIndex])*(1/similarity(userData[userIndex], userData[neighborIndex]))
                    #ratingCount += similarity(userData[userIndex], userData[neighborIndex])
            predictions[userIndex][itemIndex] = ratingSum

print predictions
for userIndex, userRow in enumerate(predictions):
    recommendations = userRow.argsort()[-5:][::-1]
    print userIndexes[userIndex],
    for recommendation in recommendations:
        if predictions[userIndex][recommendation] != -1:
            print recommendation,
    print \


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

    print 'AGGLOMERATIVE CORRELATION'
    for nClusters in range(2,11):
        userClusters = agglomerative(userData,nClusters,"correlation")
        #print ratingDensity(nClusters, userClusters, userItemMatrix)
        print silhouetteScore(userData, userClusters,"correlation")
    print '============================'
