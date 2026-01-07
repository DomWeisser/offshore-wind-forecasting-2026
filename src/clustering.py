import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import normalize


def cluster_training_latent_space(train_latent_features, k_clusters=6, method='agglomerative'):
    print(f"Clustering Training periods")
    print(f"  Training latent features Shape: {train_latent_features.shape}")
    
    # Normalize for cosine approximation
    train_latent_norm = normalise_latents(train_latent_features)
    
    if method == 'agglomerative':
        clustering_model = AgglomerativeClustering(n_clusters=k_clusters, linkage='ward', metric='euclidean')
        train_cluster_labels = clustering_model.fit_predict(train_latent_norm)
    elif method == 'gmm':
        gmm = GaussianMixture(n_components=k_clusters, covariance_type='diag', random_state=42, n_init=5)
        train_cluster_labels = gmm.fit_predict(train_latent_norm)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    unique_clusters, counts = np.unique(train_cluster_labels, return_counts=True)
    min_size = 10  # Threshold
    
    # Merge small clusters
    for cluster_id in list(unique_clusters):  # Copy to avoid modification during iteration
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
            print(f"Merged small cluster {cluster_id} ({counts[idx]} samples) into {nearest_cluster}")
    
    # Recompute after merging
    unique_clusters, counts = np.unique(train_cluster_labels, return_counts=True)
    for cluster_id, size in zip(unique_clusters, counts):
        print(f"    Cluster {cluster_id}: {size} train periods")
    
    return train_cluster_labels

def normalise_latents(latents):
    return normalize(latents, norm='l2')


def assign_test_clusters(train_latent_features, test_latent_features, train_cluster_labels):
    print(f"Assigning test periods to clusters")
    print(f"  Test latent features Shape: {test_latent_features.shape}")
    
    train_norm = normalise_latents(train_latent_features)
    test_norm = normalise_latents(test_latent_features)
    
    centroid_classifier = NearestCentroid(metric='euclidean')
    centroid_classifier.fit(train_norm, train_cluster_labels)
    test_cluster_labels = centroid_classifier.predict(test_norm)
    
    unique_clusters = np.unique(train_cluster_labels)
    for cluster_id in unique_clusters:
        test_count = np.sum(test_cluster_labels == cluster_id)
        print(f"    Cluster {cluster_id}: {test_count} test periods")
    
    return test_cluster_labels


def evaluate_clustering_quality_vae(train_latent_features, train_cluster_labels, test_cluster_labels, train_period_info, test_period_info, train_timeseries=None):
    train_unique_clusters, train_counts = np.unique(train_cluster_labels, return_counts=True)
    test_counts = np.array([np.sum(test_cluster_labels == c) for c in train_unique_clusters])
    
    # 1. Internal Clustering Metrics (on latent features)
    silhouette = silhouette_score(train_latent_features, train_cluster_labels) if len(train_unique_clusters) > 1 else 0
    davies_bouldin = davies_bouldin_score(train_latent_features, train_cluster_labels) if len(train_unique_clusters) > 1 else 2.0  # High DB is bad
    ch_score = calinski_harabasz_score(train_latent_features, train_cluster_labels) if len(train_unique_clusters) > 1 else 0
    
    # Normalize for scoring (silhouette already -1 to 1, cap others)
    db_normalized = max(0, (2.0 - davies_bouldin) / 2.0)
    ch_normalized = min(1, np.log1p(ch_score) / np.log1p(10000))  # Rough norm for typical ranges
    
    # 2. Meteorological Separation (between-cluster differences in weather features)
    # Use F-statistic (ANOVA) on key features from period_info to measure pattern distinction
    features = ['mean_wind', 'std_wind', 'std_power']  # Focus on wind variability for patterns like storms/calm
    f_stats = []
    for feature in features:
        feature_groups = []
        for cluster_id in train_unique_clusters:
            cluster_indices = np.where(train_cluster_labels == cluster_id)[0]
            feature_values = [train_period_info[i][feature] for i in cluster_indices if feature in train_period_info[i]]
            if len(feature_values) > 1:  # Need variance for ANOVA
                feature_groups.append(np.array(feature_values))
        if len(feature_groups) > 1 and all(len(g) > 1 for g in feature_groups):
            f_stat, _ = f_oneway(*feature_groups)
            f_stats.append(f_stat)
    meteo_separation = np.mean(f_stats) if f_stats else 0
    meteo_normalized = min(1, np.log1p(meteo_separation) / 10)  # Normalize for scoring
    
    # 3. Temporal Coherence (intra-cluster time series similarity)
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
    
    # 4. Distribution Consistency (train-test label proportions, for stability)
    if len(train_unique_clusters) >= 2:
        train_props = train_counts / np.sum(train_counts)
        test_props = test_counts / np.sum(test_counts)
        m = (train_props + test_props) / 2
        js_div = 0.5 * np.sum(train_props * np.log(train_props / (m + 1e-10) + 1e-10)) + \
                 0.5 * np.sum(test_props * np.log(test_props / (m + 1e-10) + 1e-10))
        distribution_consistency = max(0, 1 - js_div)
    else:
        distribution_consistency = 0
    
    # 5. Quality Score: Weighted sum emphasizing separation (internal + meteo)
    # Weights: Internal sep (sil:0.2, db:0.2, ch:0.2), Meteo sep (0.25), Temporal (0.1), Consistency (0.05)
    weights = [0.2, 0.2, 0.2, 0.25, 0.1, 0.05]
    components = [max(0, silhouette), db_normalized, ch_normalized, meteo_normalized, max(0, temporal_coherence), distribution_consistency]
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

def evaluate_forecasting_viability(train_cluster_labels, test_cluster_labels):
    
    unique_clusters = np.unique(train_cluster_labels)
    
    # Get cluster sizes
    train_counts = []
    test_counts = []
    
    for cluster_id in unique_clusters:
        train_count = np.sum(train_cluster_labels == cluster_id)
        test_count = np.sum(test_cluster_labels == cluster_id)
        train_counts.append(train_count)
        test_counts.append(test_count)
    
    train_counts = np.array(train_counts)
    test_counts = np.array(test_counts)
    
    # Training data sufficiency (for GP training)
    min_train_size = np.min(train_counts)
    train_sufficiency = min(min_train_size / 15, 1.0)  # Need 15+ periods minimum
    
    # Test data sufficiency (for evaluation)
    min_test_size = np.min(test_counts)
    test_sufficiency = min(min_test_size / 5, 1.0)   # Need 5+ test periods minimum
    
    # Distribution consistency (train vs test)
    train_props = train_counts / np.sum(train_counts)
    test_props = test_counts / np.sum(test_counts)
    
    # Jensen-Shannon divergence for distribution similarity
    def js_divergence(p, q):
        m = (p + q) / 2
        return 0.5 * np.sum(p * np.log(p/m + 1e-10)) + 0.5 * np.sum(q * np.log(q/m + 1e-10))
    
    js_div = js_divergence(train_props, test_props)
    distribution_consistency = max(0, 1 - js_div)  # Higher is better
    
    # Overall viability score
    viability_score = (
        train_sufficiency * 0.4 +      # Can we train reliable GPs?
        test_sufficiency * 0.3 +       # Can we evaluate them?
        distribution_consistency * 0.3  # Are train/test distributions similar?
    )
    
    return {
        'train_sufficiency': train_sufficiency,
        'test_sufficiency': test_sufficiency,
        'distribution_consistency': distribution_consistency,
        'min_train_size': min_train_size,
        'min_test_size': min_test_size,
        'viability_score': viability_score
    }
