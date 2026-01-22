
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, silhouette_score, calinski_harabasz_score
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd




def plot_clusters(best_combination, save_prefix=None):

    train_ts = best_combination['train_normalised_full']
    test_ts = best_combination['test_normalised_full']
    all_ts = np.vstack([train_ts, test_ts])

    train_labels = best_combination['train_cluster_labels']
    test_labels = best_combination['test_cluster_labels']
    all_labels = np.concatenate([train_labels, test_labels])

    scalers = best_combination['scalers_full']

    idx_wind = 1
    idx_sin = 13
    idx_cos = 14


    # Plot 1: Wind Speed Distribution

    raw_wind = all_ts[:, idx_wind, :]
    if scalers.get('Windspeed'):
        s = raw_wind.shape
        raw_wind = scalers['Windspeed'].inverse_transform(raw_wind.reshape(-1, 1)).reshape(s)

    mean_wind_per_sample = np.mean(raw_wind, axis=1)
    df_plot = pd.DataFrame({
        'Period-Average Wind Speed (m/s)': mean_wind_per_sample,
        'Cluster': all_labels
    })

    publication_palette = sns.color_palette("deep")
    fig1, ax1 = plt.subplots(figsize=(20, 16))

    sns.kdeplot(data=df_plot, x="Period-Average Wind Speed (m/s)", hue="Cluster",
                fill=True, common_norm=False, palette=publication_palette,
                alpha=0.25, linewidth=2.5, ax=ax1)

    ax1.set_title("Distribution of Mean Wind Speed by Cluster", fontsize=40, y=1.02)
    ax1.set_xlabel("Mean Wind Speed (m/s) over Time Period", fontsize=38)
    ax1.set_ylabel("Density Probability", fontsize=38)
    ax1.tick_params(axis='both', which='major', labelsize=34)
    plt.ylim(0, 0.2)
    ax1.grid(True, which='major', linestyle='--', alpha=0.5)
    sns.despine(ax=ax1, offset=5, trim=True)
    sns.move_legend(ax1, "upper right", title='Clusters', fontsize=40, frameon=True, shadow=False)
    leg = ax1.get_legend()
    leg.get_title().set_fontsize(40)

    if save_prefix:
        save_name_1 = f"{save_prefix}_wind_speed_distribution.png"
        plt.savefig(save_name_1, dpi=300)
        print(f"Saved Wind Speed plot to: {save_name_1}")
    plt.show()


    # Plot 2: Wind Direction Roses

    raw_sin = all_ts[:, idx_sin, :]
    raw_cos = all_ts[:, idx_cos, :]
    avg_sin = np.mean(raw_sin, axis=1)
    avg_cos = np.mean(raw_cos, axis=1)
    angles_rad = np.arctan2(avg_sin, avg_cos)

    unique_clusters = np.sort(np.unique(all_labels))
    n_clusters = len(unique_clusters)
    cols = 4
    rows = int(np.ceil(n_clusters / cols))

    fig2 = plt.figure(figsize=(20, 5 * rows))

    for i, cluster_id in enumerate(unique_clusters):
        ax = fig2.add_subplot(rows, cols, i+1, projection='polar')
        mask = all_labels == cluster_id
        cluster_angles = angles_rad[mask]

        counts, bin_edges = np.histogram(cluster_angles, bins=16, range=(-np.pi, np.pi))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = (2*np.pi) / 16
        color = publication_palette[i % len(publication_palette)]

        ax.bar(bin_centers, counts, width=width, bottom=0.0, color=color, alpha=0.7, edgecolor='white')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"Cluster {cluster_id}\n(n={np.sum(mask)})", fontweight='bold', y=1.1)
        ax.set_yticks([])

        mean_angle_cluster = np.arctan2(np.mean(np.sin(cluster_angles)), np.mean(np.cos(cluster_angles)))
        ax.arrow(mean_angle_cluster, 0, 0, np.max(counts)*0.8,
                color='black', width=0.05, head_width=0.2, alpha=0.8, zorder=10)

    plt.suptitle("Wind Direction by Cluster", fontsize=16, y=1.02)
    plt.tight_layout()

    if save_prefix:
        save_name_2 = f"{save_prefix}_wind_roses.png"
        plt.savefig(save_name_2, dpi=300)
        print(f"Saved Wind Rose plot to: {save_name_2}")
    plt.show()


def plot_latent_clusters(best_combination, save_path=None):

    sns.set_theme(style="white", context="paper", font_scale=1.2)

    latent_matrix = best_combination['train_latent']
    labels = best_combination['train_cluster_labels']

    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=50, random_state=42, init='pca', learning_rate='auto')
    z_embedded = tsne.fit_transform(latent_matrix)

    df_plot = pd.DataFrame({'x': z_embedded[:, 0], 'y': z_embedded[:, 1], 'Cluster': labels})
    publication_palette = sns.color_palette("deep")

    plt.figure(figsize=(20, 16))

    sns.kdeplot(data=df_plot, x='x', y='y', hue='Cluster',
                fill=True, palette=publication_palette, alpha=0.2,
                thresh=0.05, levels=2, legend=False, linewidth=0)

    sns.kdeplot(data=df_plot, x='x', y='y', hue='Cluster',
                fill=False, palette=publication_palette, alpha=0.7,
                thresh=0.05, levels=2, legend=False, linewidths=2.5)

    sns.scatterplot(data=df_plot, x='x', y='y', hue='Cluster',
                   palette=publication_palette, alpha=0.5, s=15, linewidth=0, legend=False)

   # plt.title("Clusters Projected in the Latent Space", fontsize=24, fontweight='bold', y=1.02)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_results(cluster_gp_models, cluster_baseline_results, save_path=None):

    clusters = sorted(set(cluster_gp_models.keys()) | set(cluster_baseline_results.keys()))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # MAE Comparison
    x_pos = np.arange(len(clusters))
    width = 0.25

    gp_maes = [cluster_gp_models[c]['metrics']['mae'] if c in cluster_gp_models else np.nan for c in clusters]
    rf_maes = [cluster_baseline_results[c]['Random Forest']['mae']
              if c in cluster_baseline_results and 'Random Forest' in cluster_baseline_results[c] else np.nan
              for c in clusters]
    lr_maes = [cluster_baseline_results[c]['Linear Regression']['mae']
              if c in cluster_baseline_results and 'Linear Regression' in cluster_baseline_results[c] else np.nan
              for c in clusters]

    axes[0,0].bar(x_pos - width, gp_maes, width, label='GP (VAE)', alpha=0.8, color='red')
    axes[0,0].bar(x_pos, rf_maes, width, label='Random Forest', alpha=0.8, color='blue')
    axes[0,0].bar(x_pos + width, lr_maes, width, label='Linear Regression', alpha=0.8, color='green')
    axes[0,0].set_xlabel('Cluster')
    axes[0,0].set_ylabel('MAE (MW)')
    axes[0,0].set_title('MAE Comparison by Cluster')
    axes[0,0].set_xticks(x_pos)
    axes[0,0].set_xticklabels([f'C{c}' for c in clusters])
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # R² Comparison
    gp_r2s = [cluster_gp_models[c]['metrics']['r2'] if c in cluster_gp_models else np.nan for c in clusters]
    rf_r2s = [cluster_baseline_results[c]['Random Forest']['r2']
             if c in cluster_baseline_results and 'Random Forest' in cluster_baseline_results[c] else np.nan
             for c in clusters]
    lr_r2s = [cluster_baseline_results[c]['Linear Regression']['r2']
             if c in cluster_baseline_results and 'Linear Regression' in cluster_baseline_results[c] else np.nan
             for c in clusters]

    axes[0,1].bar(x_pos - width, gp_r2s, width, label='GP (VAE)', alpha=0.8, color='red')
    axes[0,1].bar(x_pos, rf_r2s, width, label='Random Forest', alpha=0.8, color='blue')
    axes[0,1].bar(x_pos + width, lr_r2s, width, label='Linear Regression', alpha=0.8, color='green')
    axes[0,1].set_xlabel('Cluster')
    axes[0,1].set_ylabel('R²')
    axes[0,1].set_title('R² Comparison by Cluster')
    axes[0,1].set_xticks(x_pos)
    axes[0,1].set_xticklabels([f'C{c}' for c in clusters])
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Best Model per Cluster
    best_models = []
    best_maes = []
    colors_best = []

    for cluster_id in clusters:
        cluster_maes = {}
        if cluster_id in cluster_gp_models:
            cluster_maes['GP'] = cluster_gp_models[cluster_id]['metrics']['mae']
        if cluster_id in cluster_baseline_results:
            if 'Random Forest' in cluster_baseline_results[cluster_id]:
                cluster_maes['RF'] = cluster_baseline_results[cluster_id]['Random Forest']['mae']
            if 'Linear Regression' in cluster_baseline_results[cluster_id]:
                cluster_maes['LR'] = cluster_baseline_results[cluster_id]['Linear Regression']['mae']

        if cluster_maes:
            best_model = min(cluster_maes, key=cluster_maes.get)
            best_models.append(best_model)
            best_maes.append(cluster_maes[best_model])
            colors_best.append('red' if best_model == 'GP' else 'blue' if best_model == 'RF' else 'green')
        else:
            best_models.append('None')
            best_maes.append(0)
            colors_best.append('gray')

    bars = axes[0,2].bar(range(len(clusters)), best_maes, color=colors_best, alpha=0.8)
    axes[0,2].set_xlabel('Cluster')
    axes[0,2].set_ylabel('Best MAE (MW)')
    axes[0,2].set_title('Best Model per Cluster')
    axes[0,2].set_xticks(range(len(clusters)))
    axes[0,2].set_xticklabels([f'C{c}' for c in clusters])

    for i, (bar, model) in enumerate(zip(bars, best_models)):
        if model != 'None':
            axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                          model, ha='center', va='bottom', fontweight='bold')

    # Predictions vs Actual for each model
    plot_idx = 3
    for model_type, color in [('GP', 'red'), ('Random Forest', 'blue'), ('Linear Regression', 'green')]:
        ax = axes[plot_idx // 3, plot_idx % 3]

        all_actual = []
        all_predicted = []

        for cluster_id in clusters:
            if model_type == 'GP' and cluster_id in cluster_gp_models:
                metrics = cluster_gp_models[cluster_id]['metrics']
                all_actual.extend(metrics.get('actual_original', []))
                all_predicted.extend(metrics.get('predictions_original', []))
            elif model_type in ['Random Forest', 'Linear Regression'] and cluster_id in cluster_baseline_results:
                if model_type in cluster_baseline_results[cluster_id]:
                    metrics = cluster_baseline_results[cluster_id][model_type]
                    all_actual.extend(metrics['actual'])
                    all_predicted.extend(metrics['predictions'])

        if all_actual:
            ax.scatter(all_actual, all_predicted, alpha=0.5, s=10, color=color)
            min_val = min(np.min(all_actual), np.min(all_predicted))
            max_val = max(np.max(all_actual), np.max(all_predicted))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)

            r2 = r2_score(all_actual, all_predicted)
            mae = mean_absolute_error(all_actual, all_predicted)

            ax.set_xlabel('Actual Power (MW)')
            ax.set_ylabel('Predicted Power (MW)')
            ax.set_title(f'{model_type}\nMAE: {mae:.1f}, R²: {r2:.3f}')
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{model_type} - No Data')

        plot_idx += 1

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



def plot_cluster_predictions(final_results, optimal_config, plot_hours=72, save_path=None):

    cluster_gp_models = final_results['cluster_gp_models']

    if not cluster_gp_models:
        print("No GP models found to plot!")
        return

    print(f"\nCreating prediction plots for {len(cluster_gp_models)} clusters...")
    print(f"Displaying first {plot_hours} hours of predictions")

    n_clusters = len(cluster_gp_models)
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12*n_cols, 9*n_rows))
    if n_clusters == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    plot_idx = 0

    for cluster_id in sorted(cluster_gp_models.keys()):
        model_info = cluster_gp_models[cluster_id]
        metrics = model_info['metrics']

        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]

        if 'actual_original' in metrics and 'predictions_original' in metrics:
            y_actual_full = metrics['actual_original']
            y_pred_full = metrics['predictions_original']
            y_std_full = metrics.get('uncertainties_original', np.zeros_like(y_pred_full))
            power_unit = 'MW'
        else:
            y_actual_full = metrics.get('actual_normalized', [])
            y_pred_full = metrics.get('predictions_normalized', [])
            y_std_full = metrics.get('uncertainties_normalized', np.zeros_like(y_pred_full))
            power_unit = 'Normalized'

        if len(y_actual_full) == 0 or len(y_pred_full) == 0:
            ax.text(0.5, 0.5, f'Cluster {cluster_id}\nNo Data',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=18, fontweight='bold')
            ax.set_title(f'Cluster {cluster_id}', fontsize=18)
            plot_idx += 1
            continue

        n_points_to_plot = min(plot_hours, len(y_actual_full))
        y_actual = y_actual_full[:n_points_to_plot]
        y_pred = y_pred_full[:n_points_to_plot]
        y_std = y_std_full[:n_points_to_plot]
        hours = np.arange(n_points_to_plot)

        ax.plot(hours, y_actual, 'b-', linewidth=3, label='Actual', alpha=0.8)
        ax.plot(hours, y_pred, 'r-', linewidth=3, label='GP Prediction', alpha=0.8)

        if np.any(y_std > 0):
            ax.fill_between(hours, y_pred - y_std, y_pred + y_std,
                          color='red', alpha=0.2, label='±σ Uncertainty')

        mae = metrics.get('mae', 0)
        r2 = metrics.get('r2', 0)
        n_test = metrics.get('n_test', 0)


        ax.set_title(f'Cluster {cluster_id}\nMAE: {mae:.1f} {power_unit}, R²: {r2:.3f}',
                    fontsize=25)
        ax.set_xlabel('Hours', fontsize=24)
        ax.set_ylabel(f'Power ({power_unit})', fontsize=24)
        ax.legend(fontsize=22, loc='best')
        ax.grid(True, alpha=0.3, linewidth=1.5)

        ax.tick_params(axis='both', which='major', labelsize=20)

        if power_unit == 'MW':
            ax.set_ylim(0, max(1000, np.max(y_actual) * 1.1))

        plot_idx += 1


    for idx in range(plot_idx, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)


    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


    print(f"\nPrediction Plot Summary:")
    print(f"{'='*50}")
    for cluster_id in sorted(cluster_gp_models.keys()):
        metrics = cluster_gp_models[cluster_id]['metrics']
        mae = metrics.get('mae', 0)
        r2 = metrics.get('r2', 0)
        n_test = metrics.get('n_test', 0)
        print(f"Cluster {cluster_id}: MAE={mae:.1f} MW, R²={r2:.3f} ({n_test} test points)")

