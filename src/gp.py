import numpy as np
import torch
import gpytorch

from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, AdditiveKernel, RBFKernel, MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, Matern, RBF
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def create_cluster_forecasting_datasets(latent_features, normalised_timeseries_data, cluster_labels, period_indices, period_info, forecast_horizon=1):
    
    unique_clusters = np.unique(cluster_labels)
    cluster_datasets = {}
    
    period_mask = np.zeros(len(cluster_labels), dtype=bool)
    valid_indices = period_indices[period_indices < len(cluster_labels)]
    period_mask[valid_indices] = True
    
    # Feature indices in full timeseries matrix
    feature_indices = {
        'power': 0,
        'windspeed': 1,
        'fsr': 2,
        'u100': 3,
        'v100': 4,
        'power_t1': 5,
        'power_t2': 6,
        'power_t3': 7,
        'windspeed_t1': 8,
        'windspeed_t2': 9,
        'windspeed_t3': 10,
        'hour_sin': 11,
        'hour_cos': 12,
        'wind_dir_sin': 13,
        'wind_dir_cos': 14,
        'month_sin': 15,
        'month_cos': 16
    }
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        combined_mask = cluster_mask & period_mask  
        selected_indices = np.where(combined_mask)[0]
        
        if len(selected_indices) < 5:
            continue

        cluster_latent = latent_features[selected_indices]
        cluster_timeseries = normalised_timeseries_data[selected_indices]
        
        X_cluster = []
        y_cluster = []
        period_idx_mapping = []
        
        for local_idx, (vae_features, period_data) in enumerate(zip(cluster_latent, cluster_timeseries)):

            original_period_idx = selected_indices[local_idx]
            
            # Extract feature arrays using indices
            power_values = period_data[feature_indices['power'], :]
            power_t1_values = period_data[feature_indices['power_t1'], :]
            power_t2_values = period_data[feature_indices['power_t2'], :]
          #  power_t3_values = period_data[feature_indices['power_t3'], :]
            windspeed_values = period_data[feature_indices['windspeed'], :]
            windspeed_t1_values = period_data[feature_indices['windspeed_t1'], :]
            windspeed_t2_values = period_data[feature_indices['windspeed_t2'], :]
          #  windspeed_t3_values = period_data[feature_indices['windspeed_t3'], :]
            
            # Cyclical features 
            hour_sin_values = period_data[feature_indices['hour_sin'], :]
            hour_cos_values = period_data[feature_indices['hour_cos'], :]
            month_sin_values = period_data[feature_indices['month_sin'], :]
            month_cos_values = period_data[feature_indices['month_cos'], :]
            wind_dir_sin_values = period_data[feature_indices['wind_dir_sin'], :]
            wind_dir_cos_values = period_data[feature_indices['wind_dir_cos'], :]
            
            # Additional weather features
            fsr_values = period_data[feature_indices['fsr'], :]
        #    u100_values = period_data[feature_indices['u100'], :]
         #   v100_values = period_data[feature_indices['v100'], :]
            
            for hour in range(len(power_values) - forecast_horizon):
                features = []
                
                # VAE latent features (8 features)
                features.extend(vae_features.tolist())
                
                # Lagged power features
                features.extend([
                    power_t1_values[hour],
                    power_t2_values[hour]
                ])
                
                # Lagged wind features
                features.extend([
                    windspeed_t1_values[hour],
                    windspeed_t2_values[hour]
                ])
                
                # Current weather conditions
                features.extend([
                    windspeed_values[hour],
                    fsr_values[hour]
                ])
                
                # Cyclical temporal features 
                features.extend([
                    hour_sin_values[hour],
                    hour_cos_values[hour],
                    month_sin_values[hour],
                    month_cos_values[hour]
                ])
                
                # Cyclical wind direction
                features.extend([
                    wind_dir_sin_values[hour],
                    wind_dir_cos_values[hour]
                ])
                
                # Target
                y_target = power_values[hour + forecast_horizon]
                
                X_cluster.append(features)
                y_cluster.append(y_target)
                period_idx_mapping.append(original_period_idx)
        
        if len(X_cluster) == 0:
            continue
        
        X_cluster = np.array(X_cluster, dtype=np.float32)
        y_cluster = np.array(y_cluster, dtype=np.float32)
        
        cluster_datasets[cluster_id] = {
            'X': X_cluster,
            'y': y_cluster,
            'n_periods': len(cluster_latent),
            'n_samples': len(X_cluster),
            'n_features': X_cluster.shape[1],
            'feature_names': [
                'latent_0', 'latent_1', 'latent_2', 'latent_3', 'latent_4', 'latent_5', 'latent_6', 'latent_7',
                'power_t1', 'power_t2',  'windspeed_t1', 'windspeed_t2', 
                'windspeed', 'fsr', 'u100', 'v100',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'wind_dir_sin', 'wind_dir_cos'
            ],
            'period_indices': selected_indices.tolist(),
            'period_idx_mapping': period_idx_mapping
        }
        
        print(f"Cluster {cluster_id}: {len(cluster_latent)} periods, {len(X_cluster)} samples, {X_cluster.shape[1]} features")
    
    return cluster_datasets


def create_kernels(n_features):

    constant_kernel = ConstantKernel(1.0, (0.1, 10.0))
    rbf_kernel = RBF(length_scale=[1.0]*n_features, length_scale_bounds=(0.1, 15.0))
    matern_kernel = Matern(length_scale=[1.0]*n_features, length_scale_bounds=(0.1, 15.0), nu=1.5)
    white_kernel = WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-6, 0.1))

    kernel = constant_kernel * (rbf_kernel + matern_kernel) + white_kernel
    
    return kernel

class GPModel(ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            AdditiveKernel(
                RBFKernel(ard_num_dims=train_x.shape[1]),
                MaternKernel(nu=1.5, ard_num_dims=train_x.shape[1])
            )
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gpytorch_model(X_train, y_train, training_iterations=100, device='cuda'):

    train_x = torch.FloatTensor(X_train.astype(np.float32)).to(device)
    train_y = torch.FloatTensor(y_train.astype(np.float32)).to(device)
    
    likelihood = GaussianLikelihood().to(device)
    model = GPModel(train_x, train_y, likelihood).to(device)
    
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        
        if torch.isnan(loss) or torch.isinf(loss):
            break
        
        loss.backward()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            break
        
        optimizer.step()
    
    return model, likelihood


def predict_gpytorch_model(model, likelihood, X_test, device='cuda', batch_size=1024):

    model.eval()
    likelihood.eval()
    
    n_samples = X_test.shape[0]
    all_means = []
    all_stds = []
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            test_x_batch = torch.FloatTensor(X_test[i:batch_end]).to(device)
            predictions = likelihood(model(test_x_batch))
            all_means.append(predictions.mean.cpu().numpy())
            all_stds.append(predictions.stddev.cpu().numpy())
    
    return np.concatenate(all_means), np.concatenate(all_stds)


def train_cluster_gp_models(train_cluster_datasets, test_cluster_datasets, train_period_info, test_period_info, scalers=None,
                           device='cuda', training_iterations=100):

    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    farm_capacities = {
        'Beatrice': 588, 'Hornsea': 1218, 'Walney': 649,
        'Baltic': 475, 'Gode': 582, 'Iles': 496
    }
    
    cluster_models = {}
    power_scaler = scalers.get('Power') if scalers else None
    total_clusters = len([c for c in train_cluster_datasets.keys() if c in test_cluster_datasets])
    cluster_idx = 0
    
    for cluster_id in sorted(train_cluster_datasets.keys()):
        if cluster_id not in test_cluster_datasets:
            continue
        
        cluster_idx += 1
        print(f"[{cluster_idx}/{total_clusters}] Training Cluster {cluster_id}...", end=" ", flush=True)
        
        train_data = train_cluster_datasets[cluster_id]
        test_data = test_cluster_datasets[cluster_id]
        
        X_train = train_data['X']
        y_train = train_data['y']
        X_test = test_data['X']
        y_test = test_data['y']
        
        max_samples = 22000
        if len(X_train) > max_samples:
            np.random.seed(42)
            subsample_idx = np.random.choice(len(X_train), max_samples, replace=False)  
            X_train = X_train[subsample_idx]
            y_train = y_train[subsample_idx]
        
        if len(X_train) < 20 or len(X_test) < 2:
            print(f"Skipped (insufficient data)")
            continue
        
        test_sample_farm_ids = np.array([test_period_info[idx]['farm_id'] 
                                         for idx in test_data['period_idx_mapping']])
        
        try:
            model, likelihood = train_gpytorch_model(X_train, y_train, training_iterations, device)
            y_pred_scaled, y_std_scaled = predict_gpytorch_model(model, likelihood, X_test, device)
            
            if power_scaler is not None:
                y_pred_original = power_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                y_test_original = power_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
                y_std_original = y_std_scaled * (power_scaler.data_max_[0] - power_scaler.data_min_[0])
                y_pred_original = np.clip(y_pred_original, 0, max(farm_capacities.values()))
                
                mae = mean_absolute_error(y_test_original, y_pred_original)
                r2 = r2_score(y_test_original, y_pred_original)
                rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
                
                farm_metrics = {}
                for farm_id in np.unique(test_sample_farm_ids):
                    farm_mask = test_sample_farm_ids == farm_id
                    n_farm_samples = np.sum(farm_mask)
                    if n_farm_samples == 0:
                        continue
            
                    farm_id_clean = farm_id.strip()
                    farm_capacity = farm_capacities.get(farm_id_clean, None)

                    if farm_capacity:
                        farm_y_test = y_test_original[farm_mask]
                        farm_y_pred = y_pred_original[farm_mask]
                        farm_mae = mean_absolute_error(farm_y_test, farm_y_pred)
                        farm_metrics[farm_id] = {
                            'mae': farm_mae,
                            'mae_pct': (farm_mae / farm_capacity) * 100,
                            'rmse': np.sqrt(mean_squared_error(farm_y_test, farm_y_pred)),
                            'r2': r2_score(farm_y_test, farm_y_pred) if n_farm_samples >= 2 else np.nan,
                            'n_samples': int(n_farm_samples),
                            'capacity': farm_capacity
                        }
                
                metrics = {
                    'mae': mae, 'r2': r2, 'rmse': rmse,
                    'farm_metrics': farm_metrics,
                    'predictions_original': y_pred_original,
                    'actual_original': y_test_original,
                    'uncertainties_original': y_std_original,
                    'n_train': len(X_train),
                    'n_test': len(X_test)
                }
                
                print(f"✓ MAE: {mae:.1f} MW, R²: {r2:.3f}")
            else:
                metrics = {
                    'mae_norm': mean_absolute_error(y_test, y_pred_scaled),
                    'r2_norm': r2_score(y_test, y_pred_scaled),
                    'n_train': len(X_train),
                    'n_test': len(X_test)
                }
                print(f"✓ (normalised metrics)")
            
            cluster_models[cluster_id] = {
                'gp_model': model,
                'likelihood': likelihood,
                'power_scaler': power_scaler,
                'metrics': metrics,
                'device': device
            }
            
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            if device == 'cuda':
                torch.cuda.empty_cache()
            continue
    
    return cluster_models


def train_cluster_baseline_models(train_cluster_datasets, test_cluster_datasets, train_period_info, test_period_info, scalers=None):
    
    farm_capacities = {
        'Beatrice': 588, 
        'Hornsea': 1218,
        'Walney': 649,
        'Baltic': 475,
        'Gode': 582,
        'Iles': 496
    }
    
    cluster_baseline_results = {}
    power_scaler = scalers.get('Power') if scalers else None
    
    # Progress tracking
    total_clusters = len([c for c in train_cluster_datasets.keys() if c in test_cluster_datasets])
    cluster_idx = 0
    
    for cluster_id in sorted(train_cluster_datasets.keys()):
        if cluster_id not in test_cluster_datasets:
            continue
        
        cluster_idx += 1
        print(f"  [{cluster_idx}/{total_clusters}] Training Cluster {cluster_id} baselines... ", end="", flush=True)
        
        train_data = train_cluster_datasets[cluster_id]
        test_data = test_cluster_datasets[cluster_id]
        
        X_train = train_data['X']
        y_train = train_data['y']
        X_test = test_data['X']
        y_test = test_data['y']
        
        if len(X_train) < 50 or len(X_test) < 10:
            print(f"Skipped (insufficient data)")
            continue
        
        test_sample_farm_ids = np.array([test_period_info[idx]['farm_id'] 
                                         for idx in test_data['period_idx_mapping']])
        cluster_results = {}
        
        try:
            for model_type, model_class in [('Random Forest', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)),
                                           ('Linear Regression', LinearRegression())]:
                model_class.fit(X_train, y_train)
                pred = model_class.predict(X_test)
                
                if power_scaler is not None:
                    pred_original = power_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
                    y_test_original = power_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
                    pred_original = np.clip(pred_original, 0, max(farm_capacities.values()))
                    
                    mae = mean_absolute_error(y_test_original, pred_original)
                    r2 = r2_score(y_test_original, pred_original)
                    rmse = np.sqrt(mean_squared_error(y_test_original, pred_original))
                    
                    farm_metrics = {}
                    for farm_id in np.unique(test_sample_farm_ids):
                        farm_mask = test_sample_farm_ids == farm_id
                        n_farm_samples = np.sum(farm_mask)
                        if n_farm_samples == 0:
                            continue
                        
                        farm_capacity = farm_capacities.get(farm_id, None)
                        if farm_capacity:
                            farm_y_test = y_test_original[farm_mask]
                            farm_y_pred = pred_original[farm_mask]
                            farm_mae = mean_absolute_error(farm_y_test, farm_y_pred)
                            farm_metrics[farm_id] = {
                                'mae': farm_mae,
                                'mae_pct': (farm_mae / farm_capacity) * 100,
                                'rmse': np.sqrt(mean_squared_error(farm_y_test, farm_y_pred)),
                                'r2': r2_score(farm_y_test, farm_y_pred) if n_farm_samples >= 2 else np.nan,
                                'n_samples': int(n_farm_samples),
                                'capacity': farm_capacity
                            }
                    
                    cluster_results[model_type] = {
                        'mae': mae, 'r2': r2, 'rmse': rmse,
                        'farm_metrics': farm_metrics,
                        'predictions': pred_original,
                        'actual': y_test_original,
                        'n_train': len(X_train),
                        'n_test': len(X_test)
                    }
            
            cluster_baseline_results[cluster_id] = cluster_results
            print(f"✓ RF: {cluster_results['Random Forest']['mae']:.1f} MW, LR: {cluster_results['Linear Regression']['mae']:.1f} MW")
        
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            continue
    
    return cluster_baseline_results


