import numpy as np
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
print matrix.shape


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
print matrix


aesUserData, aesUserIndexes = readAesUserData()
aesUserData = scaleData(aesUserData)
aesUserClusters = agglomerative(aesUserData,2,"euclidean")
aesUserSimilarityMatrix = np.empty((aesUserData.shape[0],aesUserData.shape[0],))
for i, u1 in enumerate(aesUserData):
    for j, u2 in enumerate(aesUserData):
        aesUserSimilarityMatrix[i][j] = similarity(u1,u2)

aesItemData, aesItemIndexes = readAesItemData()
aesItemClusters = kmeans(aesItemData,4)
aesUserItemMatrix = readAesUserItemMatrix()
for row in aesUserItemMatrix:
    print row
aesPredctions = np.empty((aesUserItemMatrix.shape[0],aesUserItemMatrix.shape[0],))
