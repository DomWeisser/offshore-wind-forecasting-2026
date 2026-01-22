import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler

def load_wind_data(filepath):

    print(f"Loading wind data from {filepath}")

    df = pd.read_csv(filepath)
    df.rename(columns={'Unnamed: 0': 'ID', 'time': 'Time'}, inplace=True)

    df['Time'] = pd.to_datetime(df['Time'], dayfirst=True, errors='coerce')
    df = df.sort_values('Time').reset_index(drop=True)

    df = df[(df['Time'] >= '2018-01-01') & (df['Time'] <= '2020-01-01')]

    df['Power'] = pd.to_numeric(df['Power'], errors='coerce')

    df['Month'] = df['Time'].dt.month
    df['Hour'] = df['Time'].dt.hour

    for lag in [1, 2, 3]:
        df[f'Power_t-{lag}'] = df['Power'].shift(lag)
        df[f'Windspeed_t-{lag}'] = df['Windspeed'].shift(lag)

    print(f"  Loaded: {len(df)} hours")
    print(f"  Power range: {df['Power'].min():.1f} - {df['Power'].max():.1f} MW")
    print(f"  Power mean: {df['Power'].mean():.1f} MW ± {df['Power'].std():.1f} MW")

    return df



def extract_timeseries(df, time_period_hours=24):

    if 'Turn_off' in df.columns:
        initial_len = len(df)
        df = df[df['Turn_off'] == 1].copy()
        print(f"  Filtered 'Turn Off' events: Dropped {initial_len - len(df)} rows")

    df = df.sort_values('Time').reset_index(drop=True)


    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

    if 'Wind_Direction' in df.columns:
        df['Wind_Direction_sin'] = np.sin(2 * np.pi * df['Wind_Direction'] / 360)
        df['Wind_Direction_cos'] = np.cos(2 * np.pi * df['Wind_Direction'] / 360)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    full_features = [
        'Power', 'Windspeed', 'Wind_Direction', 'fsr', 'u100', 'v100',
        'Power_t-1', 'Power_t-2', 'Power_t-3',
        'Windspeed_t-1', 'Windspeed_t-2', 'Windspeed_t-3',
        'Hour_sin', 'Hour_cos', 'Wind_Direction_sin', 'Wind_Direction_cos',
        'Month_sin', 'Month_cos'
    ]

    vae_features = [
        'Windspeed', 'fsr', 'u100', 'v100',
        'Wind_Direction_sin', 'Wind_Direction_cos'
    ]

    for feature in full_features:
        if feature not in df.columns:
            if not (feature.endswith('_sin') or feature.endswith('_cos')):
                df[feature] = 0

    valid_periods = []

    expected_duration = pd.Timedelta(hours=time_period_hours - 1)


    for i in range(0, len(df) - time_period_hours + 1, time_period_hours):

        start_time = df['Time'].iloc[i]
        end_time = df['Time'].iloc[i + time_period_hours - 1]

        if (end_time - start_time) == expected_duration:
            valid_periods.append(i)

    n_periods = len(valid_periods)

    if n_periods == 0:
        print(f"  Warning: No continuous {time_period_hours}-hour periods found.")
        return np.array([]), np.array([]), []


    n_full_features = len(full_features)
    n_vae_features = len(vae_features)

    timeseries_full = np.zeros((n_periods, n_full_features, time_period_hours))
    timeseries_vae = np.zeros((n_periods, n_vae_features, time_period_hours))

    full_data = df[full_features].values
    vae_data = df[vae_features].values

    for idx, start_idx in enumerate(valid_periods):
        timeseries_full[idx] = full_data[start_idx:start_idx+time_period_hours].T
        timeseries_vae[idx] = vae_data[start_idx:start_idx+time_period_hours].T

    # Create period info
    period_info = []
    farm_id = df['farm_id'].iloc[0] if 'farm_id' in df.columns else 'Unknown'

    for idx, start_idx in enumerate(valid_periods):
        end_idx = start_idx + time_period_hours
        period_slice = df.iloc[start_idx:end_idx]

        period_info.append({
            'period_id': idx,
            'start_time': period_slice['Time'].iloc[0],
            'end_time': period_slice['Time'].iloc[-1],
            'mean_power': period_slice['Power'].mean(),
            'std_power': period_slice['Power'].std(),
            'mean_wind': period_slice['Windspeed'].mean(),
            'std_wind': period_slice['Windspeed'].std(),
            'month': period_slice['Month'].iloc[0],
            'farm_id': farm_id
        })

    skipped = (len(df) // time_period_hours) - n_periods
    print(f"  Extracted: {n_periods} periods (skipped {skipped} discontinuous blocks)")
    print(f"  Full: {timeseries_full.shape}, VAE: {timeseries_vae.shape}")

    return timeseries_full, timeseries_vae, period_info


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

def normalise_timeseries(train_timeseries, test_timeseries, feature_names, log_features=['fsr']):

    scalers = {}
    log_transforms = {}
    train_normalised = train_timeseries.copy().astype(np.float32)
    test_normalised = test_timeseries.copy().astype(np.float32)

    cyclical_features = ['sin', 'cos']

    for ch in range(train_timeseries.shape[1]):

        channel_name = feature_names[ch] if ch < len(feature_names) else f'Channel_{ch}'

        if any(cyc in channel_name for cyc in cyclical_features):
            continue


        train_channel_data = train_timeseries[:, ch, :].flatten()
        test_channel_data = test_timeseries[:, ch, :].flatten()

        if channel_name in log_features:
            train_channel_data = np.log(train_channel_data)
            test_channel_data = np.log(test_channel_data)
            log_transforms[channel_name] = True
        else:
            log_transforms[channel_name] = False

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(train_channel_data.reshape(-1, 1))

        train_scaled = scaler.transform(train_channel_data.reshape(-1, 1)).flatten()
        train_normalised[:, ch, :] = train_scaled.reshape(train_timeseries.shape[0], -1)

        test_scaled = scaler.transform(test_channel_data.reshape(-1, 1)).flatten()
        test_normalised[:, ch, :] = test_scaled.reshape(test_timeseries.shape[0], -1)

        scalers[channel_name] = scaler

    return train_normalised, test_normalised, scalers, log_transforms

