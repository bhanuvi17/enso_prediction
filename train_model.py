import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import RootMeanSquaredError
import joblib
import os

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """Frame a time series as a supervised learning dataset"""
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names.append(f'var1(t-{i})')
    
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names.append('var1(t)')
        else:
            names.append(f'var1(t+{i})')
    
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    if dropnan:
        agg.dropna(inplace=True)
    
    return agg

def classify_enso(oni_values):
    """Classify ONI values into ENSO categories"""
    conditions = [
        oni_values >= 0.5,   # El Niño
        oni_values <= -0.5,  # La Niña
    ]
    choices = ['El Niño', 'La Niña']
    return np.select(conditions, choices, default='Neutral')

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics"""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    r2 = r2_score(y_true_flat, y_pred_flat)
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / np.where(y_true_flat != 0, y_true_flat, 1e-8))) * 100
    
    return {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'MAPE': mape}

def train_enso_model(data_path='ENSO.csv'):
    """Main training function"""
    print("Loading and preprocessing data...")
    
    df_enso = pd.read_csv(data_path, parse_dates=[0])
    df_enso.set_index('Date', inplace=True)
    
    n_in = 12  
    n_out = 3  
    n_features = 1
    
    n_total = len(df_enso)
    train_end = int(0.8 * n_total)
    valid_end = int(0.9 * n_total)
    
    print(f"Train period: {df_enso.index[0]} to {df_enso.index[train_end-1]}")
    print(f"Validation period: {df_enso.index[train_end]} to {df_enso.index[valid_end-1]}")
    print(f"Test period: {df_enso.index[valid_end]} to {df_enso.index[-1]}")
    
    train_data = df_enso['ONI'][:train_end]
    valid_data = df_enso['ONI'][:valid_end]
    test_data = df_enso['ONI'][:n_total]
    
    train_supervised = series_to_supervised(train_data, n_in, n_out)
    valid_supervised = series_to_supervised(valid_data, n_in, n_out)
    test_supervised = series_to_supervised(test_data, n_in, n_out)
    
    def extract_features_targets(data, n_in, n_out, start_idx=0, end_idx=None):
        if end_idx is None:
            end_idx = len(data)
        data_subset = data.iloc[start_idx:end_idx]
        X = data_subset.iloc[:, :n_in].values
        y = data_subset.iloc[:, n_in:n_in+n_out].values
        return X, y
    
    train_data_end = len(train_supervised)
    valid_start = train_end - n_in
    valid_data_end = len(valid_supervised) - (len(valid_supervised) - (valid_end - n_in))
    test_start = valid_end - n_in
    test_data_end = len(test_supervised)
    
    X_train, y_train = extract_features_targets(train_supervised, n_in, n_out, 0, train_data_end)
    X_valid, y_valid = extract_features_targets(valid_supervised, n_in, n_out, valid_start, valid_data_end)
    X_test, y_test = extract_features_targets(test_supervised, n_in, n_out, test_start, test_data_end)
    
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    
    X_scaler.fit(X_train)
    y_scaler.fit(y_train)
    
    X_train_scaled = X_scaler.transform(X_train)
    y_train_scaled = y_scaler.transform(y_train)
    X_valid_scaled = X_scaler.transform(X_valid)
    y_valid_scaled = y_scaler.transform(y_valid)
    X_test_scaled = X_scaler.transform(X_test)
    y_test_scaled = y_scaler.transform(y_test)
    
    X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], n_in, n_features)
    X_valid_scaled = X_valid_scaled.reshape(X_valid_scaled.shape[0], n_in, n_features)
    X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], n_in, n_features)
    
    print("Building LSTM model...")
    model = Sequential(name='lstm_enso')
    model.add(LSTM(64, input_shape=(n_in, n_features), return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_out))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae', RootMeanSquaredError()])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)
    
    print("Training model...")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_valid_scaled, y_valid_scaled),
        epochs=100,
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        shuffle=False,
        verbose=1
    )
    
    print("Evaluating model...")
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_test_scaled)
    
    metrics = calculate_metrics(y_true, y_pred)
    
    y_true_class = classify_enso(y_true[:, 0])
    y_pred_class = classify_enso(y_pred[:, 0])
    accuracy = accuracy_score(y_true_class, y_pred_class)
    
    print(f"\nModel Performance:")
    print(f"Classification Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("Saving model and components...")
    model.save('models/lstm_enso_model.keras')
    joblib.dump(X_scaler, 'models/X_scaler.pkl')
    joblib.dump(y_scaler, 'models/y_scaler.pkl')
    
    model_info = {
        'n_in': n_in,
        'n_out': n_out,
        'n_features': n_features,
        'train_end': train_end,
        'valid_end': valid_end,
        'metrics': metrics,
        'accuracy': accuracy,
        'test_dates': df_enso.index[valid_end:valid_end + len(y_true)],
        'y_true': y_true,
        'y_pred': y_pred
    }
    
    joblib.dump(model_info, 'models/model_info.pkl')
    
    df_enso.to_csv('models/enso_data.csv')
    
    print("Training completed successfully!")
    print("Files saved:")
    print("- models/lstm_enso_model.keras")
    print("- models/X_scaler.pkl")
    print("- models/y_scaler.pkl")
    print("- models/model_info.pkl")
    print("- models/enso_data.csv")
    
    return model, X_scaler, y_scaler, model_info

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    
    train_enso_model('ENSO.csv')