import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def read_data(file):
	"""
	데이터를 불러온다
	:param file: 파일의 경로
	:return: 입력받은 데이터
	"""
	data = pd.read_csv(file, header=0) #header=0 인 이유는 Column명이 없는 데이터기때문임
	return data
  
def RandomForestRegression(data):
	"""
	랜덤포레스트 회귀분석
	:param data: 입력받은 데이터
	"""
	traindata = data
	model = RandomForestRegressor()
	traindata =traindata.values
	X = traindata[:, :8]
	Y = traindata[:, 8:]
	model.fit(X,Y)
	pred = model.predict(np.array([[129, 63, 217, 152, 263, 318, 280, 275]]))
	print("Prediction:", pred)
  
def main(input_file):
	data = read_data(input_file)
	randomforestreg(data)
  
if __name__ == '__main__':
	input_file = "data_input.csv"
	main(input_file)
