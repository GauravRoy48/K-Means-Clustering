##################################################################################
# Creator     : Gaurav Roy
# Date        : 18 May 2019
# Description : The code performs K-Means Clustering on the Mall_Customers.csv.
##################################################################################

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,3:5].values

# Using elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

'''
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()
# From graph, we can see that optimal number of clusters is: 5
'''

# Applying KMeans to the dataset
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
Y_kmeans = kmeans.fit_predict(X)

# Visualize the clusters
# Plotting with generic cluster labels
'''
# Plotting the datapoints according to the clusters they belong in
plt.scatter(X[Y_kmeans==0, 0], X[Y_kmeans==0, 1], s=100, c='red', label='Cluster 1', edgecolors='black')
plt.scatter(X[Y_kmeans==1, 0], X[Y_kmeans==1, 1], s=100, c='blue', label='Cluster 2', edgecolors='black')
plt.scatter(X[Y_kmeans==2, 0], X[Y_kmeans==2, 1], s=100, c='green', label='Cluster 3', edgecolors='black')
plt.scatter(X[Y_kmeans==3, 0], X[Y_kmeans==3, 1], s=100, c='cyan', label='Cluster 4', edgecolors='black')
plt.scatter(X[Y_kmeans==4, 0], X[Y_kmeans==4, 1], s=100, c='magenta', label='Cluster 5', edgecolors='black')

# Plotting the cluster centroids
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1] , s=300, c='yellow', label='Centroids', edgecolors='black')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
'''


# Renaming the cluster names based on their properties

# Plotting the datapoints according to the clusters they belong in

# Careful clients: LOW Spending score but HIGH Salary
plt.scatter(X[Y_kmeans==0, 0], X[Y_kmeans==0, 1], s=100, c='red', label='Careful', edgecolors='black')
# Standard clients: AVERAGE Spending score and AVERAGE Salary
plt.scatter(X[Y_kmeans==1, 0], X[Y_kmeans==1, 1], s=100, c='blue', label='Standard', edgecolors='black')
# Target clients: HIGH Spending score and HIGH salary
plt.scatter(X[Y_kmeans==2, 0], X[Y_kmeans==2, 1], s=100, c='green', label='Target', edgecolors='black')
# Careless clients: HIGH Spending score but LOW Salary
plt.scatter(X[Y_kmeans==3, 0], X[Y_kmeans==3, 1], s=100, c='cyan', label='Careless', edgecolors='black')
# Sensible clients: LOW Spending score and LOW Salary
plt.scatter(X[Y_kmeans==4, 0], X[Y_kmeans==4, 1], s=100, c='magenta', label='Sensible', edgecolors='black')

# Plotting the cluster centroids
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1] , s=300, c='yellow', label='Centroids', edgecolors='black')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()