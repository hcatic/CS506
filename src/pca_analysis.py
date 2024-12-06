import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def run_pca(df, numerical_features, categorical_features):
    """Run PCA and plot the first two principal components."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )

    pca_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('pca', PCA(n_components=2))
    ])

    pca_result = pca_pipeline.fit_transform(df)
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    plt.title("PCA of Airbnb Listings in NYC")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.show()

    pca_model = pca_pipeline.named_steps['pca']
    explained_variance = pca_model.explained_variance_ratio_
    print("Explained variance by each component:")
    print(f"Principal Component 1: {explained_variance[0]:.2f}")
    print(f"Principal Component 2: {explained_variance[1]:.2f}")
