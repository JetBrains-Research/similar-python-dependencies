from src.cluster_analysis.clustering import cluster_vectors, visualize_clusters, analyze_pilgrims
from src.predictions.jaccard import jaccard_distance
from src.predictions.predict_dependencies import print_closest, print_libraries
from src.preprocessing.preprocess import years_requirements
from src.preprocessing.svd import train_svd

if __name__ == "__main__":
    jaccard_distance()  # Takes a long time to pre-calculate Jaccard distances
    train_svd(libraries=False)  # Train direct project embeddings
    train_svd(libraries=True)  # Train libraries embeddings + projects as their average
    print_closest(mode="repos_direct", name="RyanBalfanz_django-sendgrid/2012-11-21",
                  amount=20, single_version=True, filter_versions=True)
    print_libraries(mode="repos_direct", name="RyanBalfanz_django-sendgrid/2012-11-21",
                    single_version=True,
                    config={"idf_power": -1, "sim_power": 1.5, "num_closest": 200}, n_suggest=10)
    cluster_vectors(input_file="models/repos_direct_embeddings.npy", algo="kmeans",
                    output_file="models/repos_direct_embeddings_clusters.txt", normalize=False)
    cluster_vectors(input_file="models/libraries_embeddings.npy", algo="kmeans",
                    output_file="models/libraries_clusters.txt")
    visualize_clusters(input_file="models/repos_direct_embeddings_clusters.txt", mode="clusters")
    visualize_clusters(input_file="models/libraries_clusters.txt", mode="clusters")
    analyze_pilgrims(input_file="models/repos_direct_embeddings.npy", n_show=10)
    years_requirements()
    pass
