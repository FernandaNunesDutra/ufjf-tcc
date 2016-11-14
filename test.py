import numpy as np
from sklearn import cluster, datasets, neighbors, metrics, preprocessing

def readCSV(fileName):
    read = np.loadtxt(fileName, delimiter=',', skiprows=1)
    return read[:, 1:]

def kmeans(data, clusters):
    kmeans = cluster.KMeans(n_clusters=clusters)
    return kmeans.fit_predict(data)

def agglomerative(data, clusters):
    agglomerative = cluster.AgglomerativeClustering(n_clusters=clusters,affinity='euclidean',linkage='average')
    return agglomerative.fit_predict(data)

def dbscan(data):
    dbscan = cluster.DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
    return dbscan.fit_predict(data)

def distance(elem1, elem2):
    return np.linalg.norm(elem1-elem2)

def intraClusterDistance(cluster, data, clustering):
    sum = 0
    for i, elem1 in enumerate(data):
        if clustering[i] == cluster:
            for j, elem2 in enumerate(data):
                if(clustering[j]==cluster):
                    sum += distance(elem1,elem2)
    return sum/2

def interClusterDistance(cluster, data, clustering):
    sum = 0
    for i, elem1 in enumerate(data):
        if clustering[i] == cluster:
            for j, elem2 in enumerate(data):
                if(clustering[j]!=cluster):
                    sum += distance(elem1,elem2)
    return sum/2

def intraInterClusterDistance(clusters, data, clustering):
    intraTotal = 0
    interTotal = 0
    for i in range(clusters):
            intraTotal += intraClusterDistance(i,data,clustering)
            interTotal += interClusterDistance(i,data,clustering)
    return interTotal/intraTotal

def calinskiHarabazScore(data, clustering):
    return metrics.calinski_harabaz_score(data,clustering)

def silhouetteScore(data,clustering):
    return metrics.silhouette_score(data,clustering)

def main():
    clusters = 4

    data = readCSV("data_aes.csv")
    min_max_scaler = preprocessing.MinMaxScaler()
    data_standardized = min_max_scaler.fit_transform(data)
    print data_standardized

    clustering = kmeans(data_standardized,clusters)
    print clustering
    print calinskiHarabazScore(data_standardized,clustering)
    print silhouetteScore(data_standardized,clustering)


if __name__ == "__main__":
    main()
