
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.spatial import ConvexHull

########################################
fA=open("C:\\Users\\ilhanp\\Desktop\\a.txt" , "r",encoding="utf-8")
fB=open("C:\\Users\\ilhanp\\Desktop\\b.txt" , 'r',encoding="utf-8")
mu, sigma = 0, 0.5 
noise1 = np.random.normal(mu, sigma, [100,2]).tolist()
noise2 = np.random.normal(mu, sigma, [100,2]).tolist()
A=fA.readlines()[4:]
B=fB.readlines()[4:]
#A.extend(noise)
#B.extend(noise)
datalistA=[]
datalistB=[]
for line in A:
    a=line.split()
    datalistA.append([float(i) for i in a])

for line in B:
    a=line.split()
    datalistB.append([float(i) for i in a]) 

data=np.asarray(datalistA+datalistB+noise1+noise2)
hullA=ConvexHull(data)
plt.scatter(data[:,0],data[:,1],c='r')
plt.xlim([-3,5])
plt.ylim([-3,5])
plt.legend(['Class1'])
plt.show()
X1=datalistA+datalistB+noise1+noise2
# 200 index value is 0 for class 1 and other 200 index value is 1 for class 2
y_true = np.concatenate([np.zeros(200), np.ones(200)], axis =0)
df = pd.DataFrame(X1).to_numpy()
##################################################
# Functions
class KMeansClustering:
    def __init__(self, X, num_clusters):
        self.K = num_clusters
        self.max_iterations = 100
        self.plot_figure = True
        self.num_examples = X.shape[0]
        self.num_features = X.shape[1]
        self.X=X

    def initialize_random_centroids(self, X):
        centroids = np.zeros((self.K, self.num_features))
        for k in range(self.K):
            centroid = X[np.random.choice(range(self.num_examples))]
            centroids[k] = centroid
        return centroids

    def create_clusters(self, X, centroids):
        clusters = [[] for _ in range(self.K)]
        for point_idx, point in enumerate(X):
            closest_centroid = np.argmin(
                np.sqrt(np.sum((point - centroids) ** 2, axis=1)))
            clusters[closest_centroid].append(point_idx)
        return clusters

    def calculate_new_centroids(self, clusters, X):
        centroids = np.zeros((self.K, self.num_features))
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(X[cluster], axis=0)
            centroids[idx] = new_centroid
        return centroids

    def predict_cluster(self, clusters, X):
        y_pred = np.zeros(self.num_examples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx
        return y_pred

    def plot_fig(self, X, y):
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.show()

    def fit(self, X):
        centroids = self.initialize_random_centroids(X)

        for it in range(self.max_iterations):
            clusters = self.create_clusters(X, centroids)
            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X)
            diff = centroids - previous_centroids
            if not diff.any():
                print("Termination criterion satisfied")
                break
        y_pred = self.predict_cluster(clusters, X)
        if self.plot_figure:
            self.plot_fig(X, y_pred)
        return y_pred

#Purity Score
def purity_score(y,y_pre):
    m = {}
    for i in range(len(y)): 
        if y_pre[i] in m: 
            m[y_pre[i]].append(i)
        else:
            m[y_pre[i]] = [i]
    tot = 0
    for i in m: 
        c = {}
        mx = 0 
        for j in m[i]: 
            if y[j] in c:
                c[y[j]] += 1
            else:
                c[y[j]] = 1
            mx = max(mx,c[y[j]])
        tot += mx
        print("purity:",tot/len(y))
    return tot/len(y)
##################################################
if __name__ == "__main__":
    num_clusters = 2
    X= df
    Kmeans = KMeansClustering(X, num_clusters)
    y_pred = Kmeans.fit(X)
"""KMeansClustering(df,2)"""
purity_score(y_true, y_pred)