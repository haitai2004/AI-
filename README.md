# 인공신경망을 사용한 주식 예측 모델

->  인공신경망를 이용하여 주식차트 분석하여 다음날 주가를 예측

->  사용 데이터셋 : US Stock Market Data & Technical Indicators

미국 주식 시장 데이터를 기반으로 다양한 기술적 지표와 주식의 가격 변화를 포함한 정보를 제공하는 데이터셋
응답 3229개
https://www.kaggle.com/datasets/nikhilkohli/us-stock-market-data-60-extracted-features?select=GOOGL.csv

**1) 사용한 라이브러리**

NumPy : 데이터 처리나 계산에 자주 사용

Pandas : CSV 파일을 처리하고 분석하는 데 사용

Matplotlib : 데이터 시각화(다양한 차트)를 할 때 사용

scikit-learn : 주어진 데이터를 특정 범위로 변환하여 머신러닝 모델의 성능을 향상시키는데 사용(스케일링)

TensorFlow & Keras : 모델 구축 및 훈련을 위해 사용(계산 효율, 모델 직관적 구축)

LSTM : 순환 신경망의 한 종류로 시간에 따른 데이터(시계열 데이터 등)의 패턴을 잘 학습하고 처리하는데 사용

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense 
```


**2)  데이터셋 **

```python
# 데이터 로드
file_path = "C:\\Users\\2004s\\Downloads\\archive\\GOOGL.csv"  # Kaggle에서 다운로드한 파일 경로를 지정
data = pd.read_csv(file_path)

# 결측치 확인
print("결측치 개수:")
print(data.isnull().sum())

# 결측치 제거
cleaned_data = data.dropna()

# 데이터 크기 확인
print(f"원본 데이터 크기: {data.shape[0]} 행")
print(f"결측치 제거 후 데이터 크기: {cleaned_data.shape[0]} 행")
```


