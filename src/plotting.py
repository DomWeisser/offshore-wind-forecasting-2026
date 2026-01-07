
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, silhouette_score, calinski_harabasz_score
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd




def plot_clusters(best_combination, save_prefix=None):

#    sns.set_theme(style="ticks", context="paper", font_scale=1.4)
    
    # --- 1. PREPARE DATA ---
    train_ts = best_combination['train_normalised_full']
    test_ts = best_combination['test_normalised_full']
    all_ts = np.vstack([train_ts, test_ts])
    
    train_labels = best_combination['train_cluster_labels']
    test_labels = best_combination['test_cluster_labels']
    all_labels = np.concatenate([train_labels, test_labels])
    
    scalers = best_combination['scalers_full']
    
    # Indices (Fixed based on your data)
    idx_wind = 1
    idx_sin = 13
    idx_cos = 14
    
    # --- 2. PLOT 1: WIND SPEED DISTRIBUTION ---
    
    # Denormalize Wind Speed
    raw_wind = all_ts[:, idx_wind, :]
    if scalers.get('Windspeed'):
        s = raw_wind.shape
        raw_wind = scalers['Windspeed'].inverse_transform(raw_wind.reshape(-1, 1)).reshape(s)

    # Calculate Mean Metrics per Sample
    mean_wind_per_sample = np.mean(raw_wind, axis=1)
    
    # Create DataFrame
    df_plot = pd.DataFrame({
        'Period-Average Wind Speed (m/s)': mean_wind_per_sample,
        'Cluster': all_labels
    })

    # Use a cleaner palette
    publication_palette = sns.color_palette("deep")

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    sns.kdeplot(
        data=df_plot, 
        x="Period-Average Wind Speed (m/s)", 
        hue="Cluster",
        fill=True, 
        common_norm=False, 
        palette=publication_palette, 
        alpha=0.25, 
        linewidth=2.5, 
        ax=ax1
    )
    
    # Formatting
    ax1.set_title("Distribution of Mean Wind Speed by Cluster", fontsize=16, fontweight='bold', y=1.02)
    ax1.set_xlabel("Mean Wind Speed (m/s) over Time Period")
    ax1.set_ylabel("Density Probability")
    plt.ylim(0, 0.25)
    ax1.grid(True, which='major', linestyle='--', alpha=0.5)
    sns.despine(ax=ax1, offset=5, trim=True)
    sns.move_legend(ax1, "upper right", title='Cluster Regime', frameon=True, shadow=False)
    
    plt.tight_layout()
    
    if save_prefix: 
        save_name_1 = f"{save_prefix}_wind_speed_distribution.png"
        plt.savefig(save_name_1, dpi=300, bbox_inches='tight')
        print(f"Saved Wind Speed plot to: {save_name_1}")
    plt.show()

    # --- 3. PLOT 2: WIND DIRECTION  ---
    
    try:
        # Extract Sin/Cos
        raw_sin = all_ts[:, idx_sin, :]
        raw_cos = all_ts[:, idx_cos, :]
        
        # Average vector per sample (get the mean direction for the whole time period)
        avg_sin = np.mean(raw_sin, axis=1)
        avg_cos = np.mean(raw_cos, axis=1)
        
        # Convert to Radians for Polar Plot (arctan2 returns -pi to pi)
        angles_rad = np.arctan2(avg_sin, avg_cos)
        
        # Setup Polar Plots Grid
        unique_clusters = np.sort(np.unique(all_labels))
        n_clusters = len(unique_clusters)
        cols = 4
        rows = int(np.ceil(n_clusters / cols))
        
        # Create a new figure for the Wind Roses
        fig2 = plt.figure(figsize=(20, 5 * rows))
        
        for i, cluster_id in enumerate(unique_clusters):
            ax = fig2.add_subplot(rows, cols, i+1, projection='polar')
            
            mask = all_labels == cluster_id
            cluster_angles = angles_rad[mask]
            
            # Histogram bins (16 sectors = 22.5 degrees each)
            counts, bin_edges = np.histogram(cluster_angles, bins=16, range=(-np.pi, np.pi))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            width = (2*np.pi) / 16
            
            # Use consistent colors
            color = publication_palette[i % len(publication_palette)]
            
            # Plot Bars
            ax.bar(bin_centers, counts, width=width, bottom=0.0, color=color, alpha=0.7, edgecolor='white')
            
            # Configure Polar Axes
            ax.set_theta_zero_location("N") # North at top
            ax.set_theta_direction(-1)      # Clockwise
            ax.set_title(f"Cluster {cluster_id}\n(n={np.sum(mask)})", fontweight='bold', y=1.1)
            ax.set_yticks([]) # Hide radial numbers
            
            # Add Arrow for Mean Direction of the cluster
            mean_angle_cluster = np.arctan2(np.mean(np.sin(cluster_angles)), np.mean(np.cos(cluster_angles)))
            ax.arrow(mean_angle_cluster, 0, 0, np.max(counts)*0.8, 
                     color='black', width=0.05, head_width=0.2, alpha=0.8, zorder=10)

        plt.suptitle("Meteorological Consistency: Wind Direction by Cluster", fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save Plot 2
        if save_prefix: 
            save_name_2 = f"{save_prefix}_wind_roses.png"
            plt.savefig(save_name_2, dpi=300, bbox_inches='tight')
            print(f"Saved Wind Rose plot to: {save_name_2}")
        plt.show()
        
    except Exception as e:
        print(f"Could not generate Wind Roses (Check feature indices for sin/cos): {e}")


def plot_latent_clusters(best_combination, save_path=None):
    
 #   sns.set_theme(style="white", context="paper", font_scale=1.2)
    
    latent_matrix = best_combination['train_latent']
    labels = best_combination['train_cluster_labels']
    
    print("Computing t-SNE...")
    # Using 'init=pca' is generally more stable than random for reproducibility
    tsne = TSNE(n_components=2, perplexity=50, random_state=42, init='pca', learning_rate='auto')
    z_embedded = tsne.fit_transform(latent_matrix)
    
    df_plot = pd.DataFrame({'x': z_embedded[:, 0], 'y': z_embedded[:, 1], 'Cluster': labels})
    
    # Use the same palette as the other plot for consistency
    publication_palette = sns.color_palette("deep")

    plt.figure(figsize=(12, 10))
    
    # --- PASS 1: Filled Areas (Transparent) ---
    # This draws the shaded "islands"
    sns.kdeplot(
        data=df_plot, x='x', y='y', hue='Cluster', 
        fill=True, palette=publication_palette, alpha=0.2, 
        thresh=0.05, levels=2, legend=False, linewidth=0
    )

    # --- PASS 2: Solid Boundaries ---
    sns.kdeplot(
        data=df_plot, x='x', y='y', hue='Cluster', 
        fill=False, palette=publication_palette, alpha=0.7, 
        thresh=0.05, levels=2, legend=False, 
        linewidths=2.5 
    )
    
    # 3. The Dots (The Texture)
    sns.scatterplot(
        data=df_plot, x='x', y='y', hue='Cluster', 
        palette=publication_palette, alpha=0.5, s=15, linewidth=0, legend=False
    )

    plt.title("Latent Space Clusters (t-SNE Projection)", fontsize=16, fontweight='bold', y=1.02)
    plt.axis('off') 
    
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_results(results):

    cluster_gp_models = results['cluster_gp_models']
    cluster_baseline_results = results['cluster_baseline_results']
    
    if not cluster_gp_models and not cluster_baseline_results:
        print("No valid results to plot!")
        return
    
    plt.figure(figsize=(16, 10))
    
    # Get all clusters
    clusters = sorted(set(cluster_gp_models.keys()) | set(cluster_baseline_results.keys()))
    
    # Plot 1: MAE Comparison
    plt.subplot(2, 3, 1)
    x_pos = np.arange(len(clusters))
    width = 0.25
    
    gp_maes = []
    rf_maes = []
    lr_maes = []
    
    for cluster_id in clusters:
        # GP MAE
        gp_maes.append(cluster_gp_models[cluster_id]['metrics']['mae'] if cluster_id in cluster_gp_models else np.nan)
        
        # RF MAE
        rf_maes.append(cluster_baseline_results[cluster_id]['Random Forest']['mae'] 
                      if cluster_id in cluster_baseline_results and 'Random Forest' in cluster_baseline_results[cluster_id] 
                      else np.nan)
        
        # LR MAE
        lr_maes.append(cluster_baseline_results[cluster_id]['Linear Regression']['mae'] 
                      if cluster_id in cluster_baseline_results and 'Linear Regression' in cluster_baseline_results[cluster_id] 
                      else np.nan)
    
    plt.bar(x_pos - width, gp_maes, width, label='GP (VAE)', alpha=0.8, color='red')
    plt.bar(x_pos, rf_maes, width, label='Random Forest', alpha=0.8, color='blue')
    plt.bar(x_pos + width, lr_maes, width, label='Linear Regression', alpha=0.8, color='green')
    
    plt.xlabel('Cluster')
    plt.ylabel('MAE (MW)')
    plt.title('MAE Comparison by Cluster')
    plt.xticks(x_pos, [f'C{c}' for c in clusters])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: R² Comparison
    plt.subplot(2, 3, 2)
    
    gp_r2s = []
    rf_r2s = []
    lr_r2s = []
    
    for cluster_id in clusters:
        gp_r2s.append(cluster_gp_models[cluster_id]['metrics']['r2'] if cluster_id in cluster_gp_models else np.nan)
        rf_r2s.append(cluster_baseline_results[cluster_id]['Random Forest']['r2'] 
                     if cluster_id in cluster_baseline_results and 'Random Forest' in cluster_baseline_results[cluster_id] 
                     else np.nan)
        lr_r2s.append(cluster_baseline_results[cluster_id]['Linear Regression']['r2'] 
                     if cluster_id in cluster_baseline_results and 'Linear Regression' in cluster_baseline_results[cluster_id] 
                     else np.nan)
    
    plt.bar(x_pos - width, gp_r2s, width, label='GP (VAE)', alpha=0.8, color='red')
    plt.bar(x_pos, rf_r2s, width, label='Random Forest', alpha=0.8, color='blue')
    plt.bar(x_pos + width, lr_r2s, width, label='Linear Regression', alpha=0.8, color='green')
    
    plt.xlabel('Cluster')
    plt.ylabel('R²')
    plt.title('R² Comparison by Cluster')
    plt.xticks(x_pos, [f'C{c}' for c in clusters])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Best Model per Cluster
    plt.subplot(2, 3, 3)
    
    best_models = []
    best_maes = []
    colors = []
    
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
            
            colors.append('red' if best_model == 'GP' else 'blue' if best_model == 'RF' else 'green')
        else:
            best_models.append('None')
            best_maes.append(0)
            colors.append('gray')
    
    bars = plt.bar(range(len(clusters)), best_maes, color=colors, alpha=0.8)
    plt.xlabel('Cluster')
    plt.ylabel('Best MAE (MW)')
    plt.title('Best Model per Cluster')
    plt.xticks(range(len(clusters)), [f'C{c}' for c in clusters])
    
    for i, (bar, model) in enumerate(zip(bars, best_models)):
        if model != 'None':
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    model, ha='center', va='bottom', fontweight='bold')
    
    # Plot 4-6: Predictions vs Actual for each model type
    plot_idx = 4
    
    for model_type, color in [('GP', 'red'), ('Random Forest', 'blue'), ('Linear Regression', 'green')]:
        plt.subplot(2, 3, plot_idx)
        
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
            plt.scatter(all_actual, all_predicted, alpha=0.5, s=10, color=color)
            min_val = min(np.min(all_actual), np.min(all_predicted))
            max_val = max(np.max(all_actual), np.max(all_predicted))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
            
            r2 = r2_score(all_actual, all_predicted)
            mae = mean_absolute_error(all_actual, all_predicted)
            
            plt.xlabel('Actual Power (MW)')
            plt.ylabel('Predicted Power (MW)')
            plt.title(f'Test Metrics - {model_type}\nMAE: {mae:.1f}, R²: {r2:.3f}')
        else:
            plt.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'{model_type} - No Data')
        
        plot_idx += 1
    
    plt.tight_layout()
   # plt.savefig("/mnt/d/Dominic/Personal/UCL/Research Project (1)/WORK FOR PAPER/Figures/GP_Source_Metrics")
    plt.show()
    
    
    print(f"\n{'='*60}")
    print("PLOTTING SUMMARY")
    print(f"{'='*60}")
    
    gp_wins = sum(1 for model in best_models if model == 'GP')
    rf_wins = sum(1 for model in best_models if model == 'RF')
    lr_wins = sum(1 for model in best_models if model == 'LR')
    
    print(f"Best model wins by cluster:")
    print(f"  GP (VAE): {gp_wins}/{len(clusters)} clusters")
    print(f"  Random Forest: {rf_wins}/{len(clusters)} clusters")
    print(f"  Linear Regression: {lr_wins}/{len(clusters)} clusters")
    
    # Overall performance
    if gp_wins >= rf_wins and gp_wins >= lr_wins:
        print(f"\n VAE-GP approach wins overall! ({gp_wins} cluster wins)")
    elif rf_wins >= lr_wins:
        print(f"\n Random Forest wins overall ({rf_wins} cluster wins)")
    else:
        print(f"\n Linear Regression wins overall ({lr_wins} cluster wins)")



def plot_cluster_predictions(final_results, optimal_config, plot_hours=72):
    
    cluster_gp_models = final_results['cluster_gp_models']
    time_period_hours = optimal_config['time_period_hours']
    
    if not cluster_gp_models:
        print("No GP models found to plot!")
        return
    
    print(f"\nCreating prediction plots for {len(cluster_gp_models)} clusters...")
    print(f"Displaying first {plot_hours} hours of predictions")

    n_clusters = len(cluster_gp_models)
    n_cols = min(3, n_clusters)  
    n_rows = (n_clusters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
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
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Cluster {cluster_id}')
            plot_idx += 1
            continue
        
        # Limit to first plot_hours hours
        n_points_to_plot = min(plot_hours, len(y_actual_full))
        y_actual = y_actual_full[:n_points_to_plot]
        y_pred = y_pred_full[:n_points_to_plot]
        y_std = y_std_full[:n_points_to_plot]
        
        hours = np.arange(n_points_to_plot)

        # Plot actual values
        ax.plot(hours, y_actual, 'b-', linewidth=2, label='Actual', alpha=0.8)
        
        # Plot predictions
        ax.plot(hours, y_pred, 'r-', linewidth=2, label='GP Prediction', alpha=0.8)
        
        # Plot uncertainty bands
        if np.any(y_std > 0):
            ax.fill_between(hours, 
                          y_pred - y_std, 
                          y_pred + y_std, 
                          color='red', alpha=0.2, label='±σ Uncertainty')

        # Get metrics (calculated on FULL dataset)
        mae = metrics.get('mae', 0)
        r2 = metrics.get('r2', 0)
        n_test = metrics.get('n_test', 0)

        ax.set_title(f'Cluster {cluster_id}\nMAE: {mae:.1f} {power_unit}, R²: {r2:.3f}\n({n_test} test samples, showing {n_points_to_plot}h)', 
                    fontsize=10)
        ax.set_xlabel('Hours')
        ax.set_ylabel(f'Power ({power_unit})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        if power_unit == 'MW':
            ax.set_ylim(0, max(1000, np.max(y_actual) * 1.1))
        
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.tight_layout()
   # plt.savefig(r"/mnt/d/Dominic/Personal/UCL/Research Project (1)/WORK FOR PAPER/Figures/GP_Source_Forecasts/forecasts_all_clusters")
    plt.show()
    
    # Print summary
    print(f"\nPrediction Plot Summary:")
    print(f"{'='*50}")
    for cluster_id in sorted(cluster_gp_models.keys()):
        metrics = cluster_gp_models[cluster_id]['metrics']
        mae = metrics.get('mae', 0)
        r2 = metrics.get('r2', 0)
        n_test = metrics.get('n_test', 0)
        print(f"Cluster {cluster_id}: MAE={mae:.1f} MW, R²={r2:.3f} ({n_test} test points)")

