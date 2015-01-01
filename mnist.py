from network import NeuralNetwork
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data import train_MNIST, test_MNIST
import yaml
import cPickle as pickle
from activations import sigmoid
import csv

if __name__ == '__main__': 
    args = yaml.load(open('init.yaml'))
    
    # preprocess data
    X, y = train_MNIST()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=args['pca'])
    X = pca.fit_transform(X)
    pickle.dump(scaler, open('MNIST_Scaler.pkl', 'wb'), 2)
    pickle.dump(pca, open('MNIST_pca.pkl', 'wb'), 2)
    
    # setup and train network
    net = NeuralNetwork(args['network']['shape'], activ=sigmoid, parameters = args['network']['parameters']) 
    net.sgd(X,y, n_epochs=args['n_epochs'], filename='MNIST_NN')
    
    # test network
    X = test_MNIST()
    X = scaler.transform(X)
    X = pca.transform(X)
    yhat = net.predict(X)
    
    # writeout predictions
    predictions_file = open('net.csv', 'wb')
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["ImageId", "Label"])
    open_file_object.writerows( zip(range(1, X.shape[0]+1), yhat) )
    predictions_file.close() 
