import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("C:/Users/Nvidia/Downloads/archive/Mall_Customers.csv")
print("Data Head:")
print(data.head())
print("\nData Info:")
print(data.info())
print("\nData Describe:")
print(data.describe())


features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)


plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
optimal_clusters =4

kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
data['Cluster'] = clusters
print("Data with Clusters:")
print(data.head())

sns.pairplot(data, hue='Cluster')
plt.show()

sns.boxplot(x='Cluster', y='Annual Income (k$)', data=data)
plt.show()

sns.boxplot(x='Cluster', y='Spending Score (1-100)', data=data)
plt.show()

numeric_data = data.select_dtypes(include=[np.number])
cluster_means = numeric_data.groupby('Cluster').mean()
print("Cluster Means:")
print(cluster_means)