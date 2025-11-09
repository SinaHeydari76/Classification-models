import numpy as np
import matplotlib.pyplot as plt
#%%

features = np.loadtxt('features.txt')
labels = np.loadtxt('labels.txt')
#%%

plt.scatter(features[labels==0,0],features[labels==0,1], label='Class 0', marker='o')
plt.scatter(features[labels==1,0],features[labels==1,1], label='Class 1', marker='x')
plt.legend()
plt.show()   
#%%
class LinearLogisticRegression:
    def __init__(self, n_in, n_out):
        self.theta = np.random.randn(n_in,n_out) 
        
    def phi(self,x):
        if len(x.shape) == 1:
            return np.concatenate(([1], x), axis=0).reshape(1,-1)
        else:
            return np.concatenate((np.ones((x.shape[0],1)), x),axis=1)     
    
    def sigmoid(self,x):
        return 1 / (1+ np.exp(-x))
    
    def predict(self, x):
        if len(x.shape) == 1:
            out = self.theta.T @ self.phi(x)
        else:
            out = self.phi(x) @ self.theta
        return self.sigmoid(out)
     
    def loss(self, labels, preds):
        labels = labels.reshape(preds.shape)
        return  - np.sum((labels * np.log(preds) + (1-labels) * np.log(1-preds)))
    
    def grad(self, labels, preds, phi_x):
        labels = labels.reshape(preds.shape)
        return np.sum((preds - labels) * phi_x,axis=0).reshape(-1,1)
    
    def hessian(self, preds, phi_x):
        R = np.eye(preds.shape[0]) * ((preds) * (1-preds))
        return phi_x.T @ R @ phi_x 
    
    
    def train(self, x_train, y_train):
        delta_loss = 1
        phi_x = self.phi(x_train)
        i= 1
        while (delta_loss > 1e-5):
            print(f'epoch: {i}')
            preds = self.predict(x_train)
            loss_start = self.loss(y_train,preds)
            
            print(f'loss_start: {loss_start}')
            g = self.grad(y_train, preds, phi_x)
            h_inv = np.linalg.inv(self.hessian(preds, phi_x))
            self.theta = self.theta - h_inv @ g
            
            preds = self.predict(x_train)
            loss_new = self.loss(y_train,preds)
            
            print(f'loss_new: {loss_new}')
            
            delta_loss = loss_start - loss_new
            print('#'*10)
#%%
llr = LinearLogisticRegression(3, 1)
llr.train(features,labels)
#%%
x1 = np.linspace(features[:,0].min(),features[:,0].max(),num=500)
x2 = np.linspace(features[:,1].min(),features[:,1].max(),num=500)
x1,x2 = np.meshgrid(x1,x2)
features_test = np.concatenate((x1.ravel().reshape(-1,1),x2.ravel().reshape(-1,1)),axis=1)
#%%
z = llr.predict(features_test)
threshold = 0.5
z = np.where(z > threshold, 1,0).reshape(x1.shape).astype(np.float32)
pmesh = plt.pcolormesh(x1,x2,z, cmap='coolwarm')
plt.scatter(features[labels==0,0],features[labels==0,1], label='Class 0', marker='o',color='white')
plt.scatter(features[labels==1,0],features[labels==1,1], label='Class 1', marker='x',color='white')
plt.xlabel('x1')
plt.xlabel('x2')
plt.legend()
plt.colorbar(pmesh)
plt.show()
#%%








