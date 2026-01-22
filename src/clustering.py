import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import normalize
from scipy.stats import f_oneway


def cluster_training_latent_space(train_latent_features, k_clusters=6):

    print(f"Clustering {train_latent_features.shape[0]} training periods into {k_clusters} clusters")

    train_latent_norm = normalize(train_latent_features, norm='l2')
    clustering_model = AgglomerativeClustering(n_clusters=k_clusters, linkage='ward', metric='euclidean')
    train_cluster_labels = clustering_model.fit_predict(train_latent_norm)

    unique_clusters, counts = np.unique(train_cluster_labels, return_counts=True)
    min_size = 10

    for cluster_id in list(unique_clusters):
        idx = np.where(unique_clusters == cluster_id)[0][0]
        if counts[idx] < min_size:
            small_mask = train_cluster_labels == cluster_id
            if np.sum(small_mask) == 0:
                continue
            small_centroid = np.mean(train_latent_norm[small_mask], axis=0)
            other_ids = unique_clusters[unique_clusters != cluster_id]
            other_centroids = [np.mean(train_latent_norm[train_cluster_labels == c], axis=0) for c in other_ids]
            distances = np.linalg.norm(np.array(other_centroids) - small_centroid, axis=1)
            nearest_cluster = other_ids[np.argmin(distances)]
            train_cluster_labels[small_mask] = nearest_cluster

    unique_clusters, counts = np.unique(train_cluster_labels, return_counts=True)
    for cluster_id, size in zip(unique_clusters, counts):
        print(f"  Cluster {cluster_id}: {size} periods")

    return train_cluster_labels

def normalise_latents(latents):
    return normalize(latents, norm='l2')


def assign_test_clusters(train_latent_features, test_latent_features, train_cluster_labels):
    print(f"Assigning test periods to clusters")
    print(f"  Test latent features Shape: {test_latent_features.shape}")

    train_norm = normalize(train_latent_features, norm='l2')
    test_norm = normalize(test_latent_features, norm='l2')

    centroid_classifier = NearestCentroid(metric='euclidean')
    centroid_classifier.fit(train_norm, train_cluster_labels)
    test_cluster_labels = centroid_classifier.predict(test_norm)

    unique_clusters = np.unique(train_cluster_labels)
    for cluster_id in unique_clusters:
        test_count = np.sum(test_cluster_labels == cluster_id)
        print(f"  Cluster {cluster_id}: {test_count} test periods")

    return test_cluster_labels


def evaluate_clustering_quality(train_latent_features, train_cluster_labels, test_cluster_labels, train_period_info, test_period_info, train_timeseries=None):

    train_unique_clusters, train_counts = np.unique(train_cluster_labels, return_counts=True)
    test_counts = np.array([np.sum(test_cluster_labels == c) for c in train_unique_clusters])

    # Internal Clustering Metrics (on latent features)
    silhouette = silhouette_score(train_latent_features, train_cluster_labels) if len(train_unique_clusters) > 1 else 0
    davies_bouldin = davies_bouldin_score(train_latent_features, train_cluster_labels) if len(train_unique_clusters) > 1 else 2.0  # High DB is bad
    ch_score = calinski_harabasz_score(train_latent_features, train_cluster_labels) if len(train_unique_clusters) > 1 else 0

    # Normalise for scoring 
    db_normalised = max(0, (2.0 - davies_bouldin) / 2.0)
    ch_normalised = min(1, np.log1p(ch_score) / np.log1p(10000))


    # Meteorological Separation (between-cluster differences in weather features)
    features = ['mean_wind', 'std_wind', 'std_power']
    f_stats = []
    for feature in features:
        feature_groups = []
        for cluster_id in train_unique_clusters:
            cluster_indices = np.where(train_cluster_labels == cluster_id)[0]
            feature_values = [train_period_info[i][feature] for i in cluster_indices if feature in train_period_info[i]]
            if len(feature_values) > 1:
                feature_groups.append(np.array(feature_values))
        if len(feature_groups) > 1 and all(len(g) > 1 for g in feature_groups):
            f_stat, _ = f_oneway(*feature_groups)
            f_stats.append(f_stat)
    meteo_separation = np.mean(f_stats) if f_stats else 0
    meteo_normalised = min(1, np.log1p(meteo_separation) / 10)


    #  Intra-cluster time series similarity
    temporal_coherence = 0
    if train_timeseries is not None:
        temporal_scores = []
        for cluster_id in train_unique_clusters:
            mask = train_cluster_labels == cluster_id
            if np.sum(mask) < 3:
                continue
            cluster_ts = train_timeseries[mask]
            correlations = []
            for i in range(min(len(cluster_ts), 10)):
                for j in range(i + 1, min(len(cluster_ts), 10)):
                    corr = np.corrcoef(cluster_ts[i].flatten(), cluster_ts[j].flatten())[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            if correlations:
                temporal_scores.append(np.mean(correlations))
        temporal_coherence = np.mean(temporal_scores) if temporal_scores else 0


    #  Distribution Consistency
    if len(train_unique_clusters) >= 2:
        train_props = train_counts / np.sum(train_counts)
        test_props = test_counts / np.sum(test_counts)
        m = (train_props + test_props) / 2
        js_div = 0.5 * np.sum(train_props * np.log(train_props / (m + 1e-10) + 1e-10)) + \
                 0.5 * np.sum(test_props * np.log(test_props / (m + 1e-10) + 1e-10))
        distribution_consistency = max(0, 1 - js_div)
    else:
        distribution_consistency = 0

    #  Quality Score
    weights = [0.2, 0.2, 0.2, 0.25, 0.1, 0.05]
    components = [max(0, silhouette), db_normalised, ch_normalised, meteo_normalised,
                 max(0, temporal_coherence), distribution_consistency]
    quality_score = np.sum([w * c for w, c in zip(weights, components)])

    return {
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': ch_score,
        'meteo_separation': meteo_separation,
        'temporal_coherence': temporal_coherence,
        'distribution_consistency': distribution_consistency,
        'n_clusters': len(train_unique_clusters),
        'quality_score': quality_score
    }

