import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer # Ganti dari iris ke breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def run_preprocessing_pipeline():
    print("Mulai proses otomatisasi preprocessing (Dataset: Breast Cancer)...")

    # 1. Load dataset Breast Cancer
    cancer = load_breast_cancer()
    df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target

    # 2. Hapus duplikat
    df_clean = df.drop_duplicates()
    print(f"Data setelah drop duplicates: {len(df_clean)} baris")

    # 3. Pisah fitur dan target
    X = df_clean[cancer.feature_names]
    y = df_clean['target']

    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) 

    # 6. Simpan scaler 
    joblib.dump(scaler, 'scaler.pkl')
    print("- Scaler baru disimpan sebagai: scaler.pkl")

    # 7. Buat DataFrame untuk training 
    df_train = pd.DataFrame(X_train_scaled, columns=cancer.feature_names)
    df_train['target'] = y_train.values
    df_train.to_csv('iris_preprocessing.csv', index=False)
    print("- Data training (scaled) disimpan sebagai: iris_preprocessing.csv")

    # 8. Simpan data test
    df_test = pd.DataFrame(X_test_scaled, columns=cancer.feature_names)
    df_test['target'] = y_test.values
    df_test.to_csv('iris_test.csv', index=False)
    print("- Data test (scaled) disimpan sebagai: iris_test.csv")

    print(f"Preprocessing selesai! Total data training: {len(df_train)} sampel dengan {len(cancer.feature_names)} fitur.")

if __name__ == "__main__":
    run_preprocessing_pipeline()
