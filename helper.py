import numpy as np


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = np.clip(1 / (1.0 + np.exp(-z)), 1e-6, 1-1e-6)
    
    return s

def save_model(w,b):

    np.savez("weights.npz",w=w,b=b)

def predict_and_save_result(X):
	
    parameters = np.load("weights.npz")
    w = parameters["w"]
    b = parameters["b"]
    
    # LINEAR -> SIGMOID
    z = np.dot(w.T,X) + b
    pred_probability = sigmoid(z)
    
    Y_prediction = pred_probability > 0.5

    # 開啟輸出的 CSV 檔案
    with open('output.csv', 'w', newline='') as csvfile:
    	import csv
    	writer = csv.writer(csvfile)
    	writer.writerow(['id', 'label'])
    	for i,j in enumerate(Y_prediction.tolist()[0]):
    		if j:
    			writer.writerow([i+1, 1])
    		else:
    			writer.writerow([i+1, 0])


