import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler

def load_wind_data(filepath):

    print(f"Loading wind data from {filepath}")
    
    df = pd.read_csv(filepath)
    df.rename(columns={'Unnamed: 0': 'ID', 'time': 'Time'}, inplace=True)

    # Check if Turn_off column exists
    if 'Turn_off' not in df.columns:
        print("  WARNING: No 'Turn_off' column found - proceeding with all data")
        turn_off_removed = 0
    else:
        initial_len = len(df)
        # Remove ALL rows where Turn_off == 0 (farm is off)
        df = df[df['Turn_off'] != 0].copy()
        turn_off_removed = initial_len - len(df)
        print(f"  Removed {turn_off_removed} hours where farm was turned off ({100*turn_off_removed/initial_len:.1f}%)")
    
    df['Power'] = pd.to_numeric(df['Power'], errors='coerce')
    power_null = df['Power'].isnull().sum()
    df = df.dropna(subset=['Power'])

    # Remove outliers (only for operating periods now)
    power_q99 = df['Power'].quantile(0.99)
    outlier_mask = (df['Power'] > power_q99 * 1.5) | (df['Power'] < 0)  
    df.loc[outlier_mask, 'Power'] = np.nan
    outliers_removed = outlier_mask.sum()

    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df = df.dropna(subset=['Time'])
    df['Month'] = df['Time'].dt.month
    df['Hour'] = df['Time'].dt.hour

    df = df[(df['Time'] >= '2018-01-01') & (df['Time'] <= '2020-01-01')]

    # lagged features
    for lag in [1, 2, 3]:
        df[f'Power_t-{lag}'] = df['Power'].shift(lag)
        df[f'Windspeed_t-{lag}'] = df['Windspeed'].shift(lag)

    df = df.dropna(subset=[f'Power_t-1', f'Power_t-2', f'Power_t-3'])

    print(f"  Loaded: {len(df)} hours")
    print(f"  Power range: {df['Power'].min():.1f} - {df['Power'].max():.1f} MW")
    print(f"  Power mean: {df['Power'].mean():.1f} MW ± {df['Power'].std():.1f} MW")
    print(f"  Null values removed: {power_null}")
    print(f"  Outliers removed: {outliers_removed}")
    print(f"  Turn-off periods removed: {turn_off_removed}")
 
    return df



def extract_timeseries(df, time_period_hours=24, require_continuous=True):
    
    n_periods = len(df) // time_period_hours
    timeseries_list = []
    period_info = []
    
    all_features = [    
        'Power', 'Windspeed', 'Wind_Direction', 'fsr', 'Hour', 'u100', 'v100', 'Month', 
        'Power_t-1', 'Power_t-2', 'Power_t-3', 'Windspeed_t-1', 'Windspeed_t-2', 'Windspeed_t-3']
    
    vae_features = ['Windspeed', 'fsr', 'Wind_Direction', 'u100', 'v100']
    
    # Check for temporal continuity
    if require_continuous:
        df['time_diff'] = df['Time'].diff().dt.total_seconds() / 3600  # Hours
        # Identify breaks (>1.5 hours gap)
        df['is_continuous'] = df['time_diff'] <= 1.5
        df.loc[df.index[0], 'is_continuous'] = True 
    
    skipped_discontinuous = 0
    
    for i in range(n_periods):
        start_idx = i * time_period_hours
        end_idx = start_idx + time_period_hours
        
        if end_idx <= len(df):
            
            # Check if this period is temporally continuous
            if require_continuous:
                period_continuous = df['is_continuous'].iloc[start_idx:end_idx].all()
                if not period_continuous:
                    skipped_discontinuous += 1
                    continue  # Skip this period
            
            period_data_full = []
            period_data_vae = []

            for feature in all_features:
                if feature in ['Hour', 'Month', 'Wind_Direction']:      
                    continue  
                    
                if feature in df.columns:
                    feature_values = df[feature].iloc[start_idx:end_idx].values
                    period_data_full.append(feature_values)
                    
                    if feature in vae_features:
                        period_data_vae.append(feature_values)
                else:
                    period_data_full.append(np.zeros(time_period_hours))

            # Add cyclical features (same as before)
            actual_hours = df['Hour'].iloc[start_idx:end_idx].values
            hour_sin = np.sin(2 * np.pi * actual_hours / 24)
            hour_cos = np.cos(2 * np.pi * actual_hours / 24)
            period_data_full.append(hour_sin)
            period_data_full.append(hour_cos)
            
            wind_directions = df['Wind_Direction'].iloc[start_idx:end_idx].values
            wind_dir_sin = np.sin(2 * np.pi * wind_directions / 360)
            wind_dir_cos = np.cos(2 * np.pi * wind_directions / 360)
            period_data_full.append(wind_dir_sin)
            period_data_full.append(wind_dir_cos)
            
            actual_months = df['Month'].iloc[start_idx:end_idx].values
            month_sin = np.sin(2 * np.pi * actual_months / 12)
            month_cos = np.cos(2 * np.pi * actual_months / 12)
            period_data_full.append(month_sin)
            period_data_full.append(month_cos)

            period_data_vae.append(wind_dir_sin)
            period_data_vae.append(wind_dir_cos)

            period_power = df['Power'].iloc[start_idx:end_idx].values
            period_wind = df['Windspeed'].iloc[start_idx:end_idx].values
            farm_id = df['farm_id'].iloc[start_idx] if 'farm_id' in df.columns else 'Unknown'

            timeseries_list.append({
                    'full': np.array(period_data_full),
                    'vae': np.array(period_data_vae)
                })

            period_info.append({
                    'period_id': i,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_time': df['Time'].iloc[start_idx] if start_idx < len(df) else None,
                    'mean_power': np.nanmean(period_power),
                    'std_power': np.nanstd(period_power),
                    'mean_wind': np.nanmean(period_wind),
                    'std_wind': np.nanstd(period_wind),
                    'month': df['Month'].iloc[start_idx] if start_idx < len(df) else None,
                    'farm_id': farm_id
                })
    
    timeseries_matrix_full = np.array([item['full'] for item in timeseries_list])
    timeseries_matrix_vae = np.array([item['vae'] for item in timeseries_list])
    
    print(f"  Extracted: {len(timeseries_list)} {time_period_hours}-hour continuous periods")
    if require_continuous:
        print(f"  Skipped {skipped_discontinuous} periods due to temporal discontinuities")
    print(f"  Full matrix shape: {timeseries_matrix_full.shape}")
    print(f"  VAE matrix shape: {timeseries_matrix_vae.shape}")
    
    return timeseries_matrix_full, timeseries_matrix_vae, period_info





def split_timeperiods(timeseries_matrix, period_info, test_ratio=0.2, random_state=42):
    
    n_periods = len(timeseries_matrix)
    
    np.random.seed(random_state)
    all_indices = np.arange(n_periods)
    np.random.shuffle(all_indices)
    
    split_point = int((1 - test_ratio) * n_periods)
    train_indices = sorted(all_indices[:split_point])
    test_indices = sorted(all_indices[split_point:])
    
    train_timeseries = timeseries_matrix[train_indices]
    test_timeseries = timeseries_matrix[test_indices]
    train_period_info = [period_info[i] for i in train_indices]
    test_period_info = [period_info[i] for i in test_indices]

    train_farm_ids = np.array([info['farm_id'] for info in train_period_info])
    test_farm_ids = np.array([info['farm_id'] for info in test_period_info])
    
    print(f"Timeseries randomly split into {len(train_indices)} train periods and {len(test_indices)} test periods")
    
    return {
        'train_timeseries': train_timeseries,
        'test_timeseries': test_timeseries,
        'train_period_info': train_period_info,
        'test_period_info': test_period_info,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'train_farm_ids': train_farm_ids,  
        'test_farm_ids': test_farm_ids     
    }


def normalise_timeseries(train_timeseries, test_timeseries):
    
    print("Normalising VAE timeseries using VAE training data statistics...")
    
    scalers = {}
    log_transforms = {}  
    train_normalised = train_timeseries.copy().astype(np.float32)
    test_normalised = test_timeseries.copy().astype(np.float32)
    
    feature_names = [
        'Windspeed', 'fsr', 'u100', 'v100', 'wind_dir_sin', 'wind_dir_cos'
        ]
    
    for ch in range(train_timeseries.shape[1]):
        channel_name = feature_names[ch] if ch < len(feature_names) else f'Channel_{ch}'
        
        if channel_name in ['wind_dir_sin', 'wind_dir_cos']:
            print(f"  Skipping normalisation for {channel_name} (cyclical feature)")
            continue
        
        train_channel_data = train_timeseries[:, ch, :].flatten()
        test_channel_data = test_timeseries[:, ch, :].flatten()
        
        # Apply log transform to fsr
        if channel_name == 'fsr':
            print(f"  Applying log transform to {channel_name}")
            train_channel_data = np.log(train_channel_data)
            test_channel_data = np.log(test_channel_data)
            log_transforms[channel_name] = True
        else:
            log_transforms[channel_name] = False
        
        # Apply MinMaxScaler
        scaler = MinMaxScaler(feature_range=(-1, 1))                                                    #Should I use MinMax Scaler or Standard Scaler?
        scaler.fit(train_channel_data.reshape(-1, 1))
        
        train_scaled = scaler.transform(train_channel_data.reshape(-1, 1)).flatten()
        train_normalised[:, ch, :] = train_scaled.reshape(train_timeseries.shape[0], -1)
        
        test_scaled = scaler.transform(test_channel_data.reshape(-1, 1)).flatten()
        test_normalised[:, ch, :] = test_scaled.reshape(test_timeseries.shape[0], -1)
        
        scalers[channel_name] = scaler
    
    print(f"  Normalised {train_timeseries.shape[1]} channels using MinMaxScaler")
    
    return train_normalised, test_normalised, scalers


def normalise_full_timeseries(train_timeseries, test_timeseries):

    print("Normalising FULL timeseries using FULL training data statistics...")
    
    scalers = {}
    log_transforms = {}  
    train_normalised = train_timeseries.copy().astype(np.float32)
    test_normalised = test_timeseries.copy().astype(np.float32)
    
    feature_names = [
        'Power', 'Windspeed', 'fsr', 'u100', 'v100', 
        'Power_t-1', 'Power_t-2', 'Power_t-3', 
        'Windspeed_t-1', 'Windspeed_t-2', 'Windspeed_t-3',
        'hour_sin', 'hour_cos', 'wind_dir_sin', 'wind_dir_cos', 'month_sin', 'month_cos'
    ]
    
    cyclical_features = ['hour_sin', 'hour_cos', 'wind_dir_sin', 'wind_dir_cos', 'month_sin', 'month_cos']
    
    for ch in range(train_timeseries.shape[1]):
        channel_name = feature_names[ch] if ch < len(feature_names) else f'Channel_{ch}'
        
        if channel_name in cyclical_features:
            print(f"  Skipping normalisation for {channel_name} (cyclical feature)")
            continue

        train_channel_data = train_timeseries[:, ch, :].flatten()
        test_channel_data = test_timeseries[:, ch, :].flatten()
        
        # Apply log transform to fsr
        if channel_name == 'fsr':
            print(f"  Applying log transform to {channel_name}")
            train_channel_data = np.log(train_channel_data)
            test_channel_data = np.log(test_channel_data)
            log_transforms[channel_name] = True
        else:
            log_transforms[channel_name] = False
        
        # Apply MinMaxScaler
        scaler = MinMaxScaler(feature_range=(-1, 1))                                                                    # Check scaler used here as well, if changed above
        scaler.fit(train_channel_data.reshape(-1, 1))
        
        train_scaled = scaler.transform(train_channel_data.reshape(-1, 1)).flatten()
        train_normalised[:, ch, :] = train_scaled.reshape(train_timeseries.shape[0], -1)
        
        test_scaled = scaler.transform(test_channel_data.reshape(-1, 1)).flatten()
        test_normalised[:, ch, :] = test_scaled.reshape(test_timeseries.shape[0], -1)
        
        scalers[channel_name] = scaler
    
    print(f"  Normalised {train_timeseries.shape[1]} channels using MinMaxScaler")
    
    return train_normalised, test_normalised, scalers, log_transforms