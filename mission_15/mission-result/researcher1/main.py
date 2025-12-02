import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
import shutil


def load_and_preprocess_data(file_path):
    """데이터 로드 및 전처리"""
    df = pd.read_csv(file_path)
    df['Extracurricular Activities'] = df['Extracurricular Activities'].apply(
        lambda x: 1 if x == 'Yes' else 0
    )
    return df


def normalize_features(df):
    """특성 정규화"""
    scaler = StandardScaler()
    X = df.drop("Performance Index", axis=1)
    y = df["Performance Index"]
    
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, scaler


def train_model(X_train, y_train):
    """모델 훈련"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_valid, y_valid):
    """모델 평가"""
    y_pred = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    return rmse


def save_model(model, scaler, model_path='./models/best_linear_model.pkl'):
    """모델 저장"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)


def copy_test_data(test_path='./data/mission15_test.csv', output_path='./models/mission15_test.csv'):
    """테스트 데이터를 models 폴더에 복사 (연구자2가 사용할 수 있도록)"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copy(test_path, output_path)
    print(f"테스트 데이터 복사 완료: {output_path}")


def main():
    print("=" * 60)
    print("모델 학습 및 파일 준비")
    print("=" * 60)
    
    # 데이터 로드 및 전처리
    train_path = "./data/mission15_train.csv"
    train_df = load_and_preprocess_data(train_path)
    
    # 데이터 정규화
    X, y, scaler = normalize_features(train_df)
    
    # 훈련/검증 데이터 분할
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 모델 훈련
    model = train_model(X_train, y_train)
    
    # 모델 평가
    rmse = evaluate_model(model, X_valid, y_valid)
    print(f"\n학습 완료 - RMSE: {rmse:.4f}")
    
    # 모델 저장
    model_path = './models/best_linear_model.pkl'
    save_model(model, scaler, model_path)
    print(f"모델 저장 완료: {model_path}")
    
    # 테스트 데이터 복사 (연구자2를 위해)
    test_path = "./data/mission15_test.csv"
    copy_test_data(test_path, './models/mission15_test.csv')
    
    print("\n" + "=" * 60)
    print("모든 작업 완료!")
    print("연구자2가 사용할 파일:")
    print("  - models/best_linear_model.pkl (모델)")
    print("  - models/mission15_test.csv (테스트 데이터)")
    print("=" * 60)


if __name__ == "__main__":
    main()