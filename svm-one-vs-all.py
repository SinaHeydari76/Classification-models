from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn import svm as SVM
#%%
mnist = fetch_openml('mnist_784')
#%%
X = mnist.data.values
Y = mnist.target.values.astype(np.int16)
#%%
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=1/7, random_state=42, shuffle=True)
#%%
scaler = StandardScaler()
scaler = scaler.fit(x_train)
#%%
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#%%
C = [0.001, 0.01, 0.1, 1]
kernels = ['linear','rbf','poly']
#%%

for kernel in kernels:
    for c in C:
        svm = SVM.SVC(kernel=kernel,C=c, decision_function_shape='ovr')
        # since training SVM models takes time, we only use the first 10k training data
        svm.fit(x_train[:10000], y_train[:10000])
        preds = svm.predict(x_test)
        print(f'SVM model with Kernel={kernel}, and C={c} ')
        print(f'Number of support vectors: {len(svm.support_)}')
        print(classification_report(y_test, preds))
    
