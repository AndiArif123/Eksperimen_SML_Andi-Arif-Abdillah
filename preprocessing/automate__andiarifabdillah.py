import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def run_preprocessing_pipeline():
    print("Mulai proses otomatisasi preprocessing...")

    # Load dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # Hapus duplikat
    df_clean = df.drop_duplicates()
    print(f"Data setelah drop duplicates: {len(df_clean)} baris")

    # Pisah fitur dan target
    X = df_clean[iris.feature_names]
    y = df_clean['target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Pakai scaler yang sama!

    # Simpan scaler untuk dipakai saat inferensi nanti
    joblib.dump(scaler, 'scaler.pkl')
    print("- Scaler disimpan sebagai: scaler.pkl")

    # Buat DataFrame untuk training (data utama yang akan dipakai model)
    df_train = pd.DataFrame(X_train_scaled, columns=iris.feature_names)
    df_train['target'] = y_train.values
    df_train.to_csv('iris_preprocessing.csv', index=False)
    print("- Data training (scaled) disimpan sebagai: iris_preprocessing.csv")

    # Opsional: simpan data test juga (bagus untuk evaluasi)
    df_test = pd.DataFrame(X_test_scaled, columns=iris.feature_names)
    df_test['target'] = y_test.values
    df_test.to_csv('iris_test.csv', index=False)
    print("- Data test (scaled) disimpan sebagai: iris_test.csv")

    print(f"Preprocessing selesai! Total data training: {len(df_train)} sampel")

if __name__ == "__main__":
    run_preprocessing_pipeline()
