import numpy as np 
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt 

def param_init():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1,b1,W2,b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    Z = np.array(Z, dtype=float)
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_propagation(W1,b1,W2,b2,X):
    A0 = X
    Z1 = W1.dot(A0) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot_encoder(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def backward_propagation(Z1,A1,Z2,A2,W2,X,Y,m):
    one_hot_Y = one_hot_encoder(Y)
    dZ2  = A2 - one_hot_Y
    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2,axis=1).reshape(-1,1)
    dZ1 = (W2.T).dot(dZ2) * deriv_ReLU(Z1)
    dW1 = (1/m) * dZ1.dot(X.T)
    db1 = (1/m) * np.sum(dZ1,axis=1).reshape(-1,1) 
    return dW1, db1, dW2, db2


def update_params(W1,b1,W2,b2, dW1,db1, dW2,db2,alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1,b1,W2,b2

def get_predictions(A2):
    return np.argmax(A2,0)

def get_accuracy(predictions, Y):
    print(predictions,Y)
    return np.sum(predictions == Y)/ Y.size


def gradient_descent(X,Y,iterations,alpha,m):
    W1,b1,W2, b2 = param_init()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1,b1,W2,b2,X)
        dW1, db1, dW2, db2 = backward_propagation(Z1,A1,Z2,A2,W2,X,Y,m)
        W1,b1,W2,b2 = update_params(W1,b1,W2,b2, dW1,db1, dW2,db2,alpha)
        if(i % 50 == 0):
            print("Iteration : ",i)
            print("Accuracy L: ",get_accuracy(get_predictions(A2),Y))
    return W1,b1,W2,b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2,X_train,Y_train):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    current_image = np.array(current_image, dtype=float)
    # plt.gray()
    plt.imshow(current_image, cmap='gray')
    plt.show()


def main():

    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

    # plot one random digital image
    #plt.title('The 8th image is a {label}'.format(label=int(y[8]))) 
    #plt.imshow(X[8,:].reshape((28,28)), cmap='gray')
    #plt.show()

    #Process data
    # reshape y to have shape (70000, 1)
    y = y.reshape(-1, 1)
    data = np.concatenate((y,X),axis=1)

    m,n = data.shape
    print(f"m = {m},n = {n}")
    data = np.array(data)
    np.random.shuffle(data)

    #Split data
    data_dev = data[0:1000,:].T
    data_train = data[1000:m,:].T
    Y_dev = data_dev[0,:]
    X_dev = data_dev[1:n,:]
    Y_train = data_train[0,:]
    X_train = data_train[1:n,:]
    # Convert string labels to integers
    Y_train = Y_train.astype(np.int64)
    Y_dev = Y_dev.astype(np.int64)
    X_dev = X_dev / 255.
    X_train = X_train / 255.

    W1,b1,W2,b2 = gradient_descent(X_train, Y_train, 500,0.1,m)

    #test Neural network
    test_prediction(3,W1,b1,W2,b2,X_train, Y_train)
    dev_predictions = make_predictions(X_dev,W1,b1,W2,b2)
    get_accuracy(dev_predictions,Y_dev)


if __name__ == '__main__':
    main()
