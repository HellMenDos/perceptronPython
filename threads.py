import numpy as np 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

data = np.array([
    [0,0,1],
    [1,1,1],
    [1,0,1]
])
out = np.array([
    [0,1,1]
]).T

np.random.seed(1)

synphW = 2 * np.random.random((3,1)) - 1
for i in range(20000):
    exmpData = data
    output = sigmoid(np.dot(exmpData,synphW))
    
    err = (out - output)
    print(err)
    abj = np.dot(exmpData.T, (err * (output * (1- output)) ))
    synphW += abj

print(out)
