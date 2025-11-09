import numpy as np
import matplotlib.pyplot as plt

features = np.loadtxt('features.txt')
labels = np.loadtxt('labels.txt')

x1 = np.linspace(features[:,0].min(), features[:,0].max(), num=500)
x2 = np.linspace(features[:,1].min(), features[:,1].max(), num=500)
x1,x2 = np.meshgrid(x1,x2)
x_test = np.concatenate((x1.reshape(-1,1), x2.reshape(-1,1)),axis=1)

tau = 5
lambda_coef = 1e-4
theta = np.random.randn(3, 1)

def cal_weights(x, x_train):
    w = np.exp(-np.sum((x - x_train) ** 2,axis=1)/(2*tau**2)).reshape(-1,1)
    w = (w - w.min()) / (w.max() - w.min())
    return w

def phi(x):
    if len(x.shape) == 1:
        return np.concatenate(([1], x), axis=0).reshape(-1,1)
    else:
        return np.concatenate((np.ones((x.shape[0],1)), x),axis=1)     

def sigmoid(x):
    out = 1 / ( 1 + np.exp(-x) )
    return out

def predict(x,theta):
    if len(x.shape) == 1:
        out = theta.T @ phi(x)
    else:
        out = phi(x) @ theta
    
    out=sigmoid(out)
    
    return out

def loss(preds, labels, w):
    labels = labels.reshape(preds.shape)
    likelihood_term = np.sum(w * (labels * np.log(preds) + (1-labels) * np.log(1-preds)))
    regularization_term = (lambda_coef/2) * (theta.T @ theta)
    loss = regularization_term.item() - likelihood_term
    return loss

def grad(preds, labels, w, phi_x):
    labels = labels.reshape(preds.shape)
    likelihood_term = np.sum(w * (preds - labels) * phi_x,axis=0)
    likelihood_term = likelihood_term.reshape(-1,1)
    regularization_term = lambda_coef * theta
    grad = regularization_term - likelihood_term
    return grad

def hessian(preds, w, phi_x):
    R = w * preds * (1- preds) * np.eye(phi_x.shape[0])
    likelihood_term = phi_x.T @ R @ phi_x    
    regularization_term = lambda_coef * np.eye(phi_x.shape[1])
    hessian = regularization_term - likelihood_term
    return hessian

def train(x_train, y_train, w):
    phi_x = phi(x_train)
    theta = np.random.randn(3, 1)
    while(True):
        preds = predict(x_train,theta)
        loss_perv = loss(preds, y_train, w)
        print(loss_perv)
        g = grad(preds, y_train, w, phi_x)
        h = hessian(preds, w, phi_x)
        h_inv = np.linalg.inv(h)
        theta_new = theta - h_inv @ g
        preds = predict(x_train,theta_new)
        
        loss_new = loss(preds, y_train, w)
        print(loss_new)
        loss_delta = loss_perv - loss_new 
        if (loss_delta > 1e-5):
            theta = theta_new
            print('weight updated.')
        else:
            break
        print('#'*20)
    return theta

z = np.zeros(x_test.shape[0])
for i,x in enumerate(x_test):
    w = cal_weights(x, features)
    theta = train(features, labels, w)
    z[i] = predict(x,theta).item()

z = np.where( z >= 0.5 , 1, 0)

#plt.scatter(x_test[z==0,0], x_test[z==0,1], label='test_points 0',color='blue')
#plt.scatter(x_test[z==1,0], x_test[z==1,1], label='test_points 1',color='red')
pmesh = plt.pcolormesh(x1,x2, z.reshape(x1.shape), cmap='coolwarm')
plt.scatter(features[labels==0,0],features[labels==0,1], label='Class 0', color='blue', marker='o',edgecolors='white')
plt.scatter(features[labels==1,0],features[labels==1,1], label='Class 1', color='red', marker='*',edgecolors='white')
plt.colorbar(pmesh)
plt.legend()
plt.show()