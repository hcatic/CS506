import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def determine_optimal_clusters(df, lat_col='lat', long_col='long', max_k=15):
    """Use the elbow method to determine the optimal number of clusters."""
    inertia = []
    K = range(1, max_k)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df[[lat_col, long_col]])
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(K, inertia, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Determining Optimal k')
    plt.grid(True)
    plt.show()

def run_kmeans_clustering(df, lat_col='lat', long_col='long', k=5):
    """Run KMeans clustering and return DataFrame with cluster assignments."""
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['location_cluster'] = kmeans.fit_predict(df[[lat_col, long_col]])
    df = pd.get_dummies(df, columns=['location_cluster'], prefix='cluster')
    df.drop([lat_col, long_col], axis=1, inplace=True)
    return df
