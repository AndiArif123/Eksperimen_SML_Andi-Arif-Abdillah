import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def run_preprocessing_pipeline():
    # Semua kode di bawah ini WAJIB menjorok ke dalam (indentasi)
    print("Mulai proses otomatisasi preprocessing...")

    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # Preprocessing
    df_clean = df.drop_duplicates()
    X = df_clean[iris.feature_names]
    y = df_clean['target']

    # Split & Scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Membentuk DataFrame hasil
    df_final = pd.DataFrame(X_train_scaled, columns=iris.feature_names)
    df_final['target'] = y_train.values

    # Menyimpan file output
    output_path = 'iris_preprocessing.csv'
    df_final.to_csv(output_path, index=False)

    print(f"- Berhasil! Data disimpan di: {output_path}")
    print(f"- Jumlah sampel: {len(df_final)}")

if __name__ == "__main__":
    run_preprocessing_pipeline()
