# 인공신경망을 사용한 주식 예측 모델
->  인공신경망를 이용하여 주식차트 분석하여 다음날 주가를 예측
->  사용 데이터셋 : US Stock Market Data & Technical Indicators

미국 주식 시장 데이터를 기반으로 다양한 기술적 지표와 주식의 가격 변화를 포함한 정보를 제공하는 데이터셋
응답 3229개
https://www.kaggle.com/datasets/nikhilkohli/us-stock-market-data-60-extracted-features?select=GOOGL.csv

**1)**
'import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense'
