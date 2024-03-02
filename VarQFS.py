from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import pennylane as qml 
from pennylane import numpy as np
from loguru import logger

num_qubits = 20

#dev = qml.device('default.qubit.torch', wires = num_qubits, shots=None) # lightning.qubit 10000 , torch_device='cuda'

device = 'lightning.qubit'
dev = qml.device(device, wires = num_qubits, shots=10000) # lightning.qubit 10000 , torch_device='cuda', default.mixed



logger.debug(f'The number of qubits used for the experiment is {num_qubits}')
logger.debug(f'The device used for the experiment is {device}')


#@qml.qnode(dev)
def ansatz(theta:list, num_qubits=10, depth=1):
    '''
    
    '''
    
    step = 0
    for _ in range(depth):
        for i in range(num_qubits):
            qml.RY(theta[i+step], wires=i)
        for i in range(num_qubits-1):
            qml.CNOT([i,i+1])
        for i in range(num_qubits):
            qml.RY(theta[i+step], wires=i)
        step += num_qubits
        #qml.RY(theta[num_qubits-1], wires=num_qubits)
    #return qml.expval(qml.PauliZ(0))


@qml.qnode(dev)
def ansatz_2(theta:list, num_qubits=10, depth=1):
    '''
    
    '''
    
    step = 0
    for _ in range(depth):
        for i in range(num_qubits):
            qml.RY(theta[i+step], wires=i)
        for i in range(num_qubits-1):
            qml.CNOT([i,i+1])
        for i in range(num_qubits):
            qml.RY(theta[i+step], wires=i)
        step += num_qubits
        #qml.RY(theta[num_qubits-1], wires=num_qubits)
    return qml.counts()

#@qml.qnode(dev)
def amplitudes(f=None, num_qubits=None):
    qml.AmplitudeEmbedding(features=f, pad_with=0.,wires=range(num_qubits),normalize=True)
    #return qml.expval(qml.PauliZ(0))


#@qml.qnode(dev, interface="torch", diff_method="backdrop")

@qml.qnode(dev, interface="autograd") # , interface="autograd"
def circuit(weights, x, num_qubits, depth):
    '''
    Parametrized circuit with data encoding (statepreparation) and layer repetition based on the weights 
    Args:
        weights: angles for the rotations (num layer, num qubits, num directions)
        x: input vector
    Return: 
        Expectation values measured on Pauli Z operators for the state 0
    '''
    # data encoding 
    amplitudes(x, num_qubits=num_qubits)

    # ansatz 
    #for W in weights:
    ansatz(weights,num_qubits=num_qubits, depth=depth)

    # measure
   
    return qml.expval(qml.PauliZ(0))



def variational_classifier(weights, x, num_qubits, depth, bias):
    '''
    Build the parametrized circuit with weights, x and bias term
    Args:
        - weights: rotation angles 
        - bias: classical term to add more freedom to the VQA
        - x: input vector/data 
    Returns: 
        - parametrized circuit with a bias term 
    '''
    return circuit(weights, x, num_qubits, depth) + bias


def square_loss(labels, predictions):
    '''
    Compute the cost function
    Args:
        - labels: Ground truth
        - predictions: Predicted values 
    Returns: 
        - Mean of the square error between labels and predictions = model's error 
    '''
    
    # We use a call to qml.math.stack to allow subtracting the arrays directly
    #print(labels, predictions)
    return np.mean((labels - qml.math.stack(predictions)) ** 2)


def accuracy(labels, predictions):
    '''
    Compute the accuracy of the model
    Args:
        - labels: Ground truth
        - predictions: Predicted values 
    Returns: 
        - accuracy
    '''
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc

def cost(weights, num_qubits, depth, bias, X, Y):
    '''
    Compute the cost of the model
    Args: 
        - weights: rotation angles 
        - bias: classical term to add more freedom to the VQA
        - X: input vector/data 
        - Y: True labels 
    Returns: 
        - Error prediction / distance 
    '''
    
    predictions = [variational_classifier(weights, x, num_qubits, depth, bias)._value.tolist() for x in X]
    #print(predictions)
    return square_loss(Y, predictions)

def cost2(weights, num_qubits, depth, bias, X, Y):
    '''
    Compute the cost of the model
    Args: 
        - weights: rotation angles 
        - bias: classical term to add more freedom to the VQA
        - X: input vector/data 
        - Y: True labels 
    Returns: 
        - Error prediction / distance 
    '''
    
    predictions = [variational_classifier(weights, x, num_qubits, depth, bias) for x in X]
    #print(predictions)
    return square_loss(Y, predictions)

def data_processing(data_name):
    
    data = pd.read_csv(data_name, sep = ' ')
    col = [
        'Status of existing checking account',
        'Duration in month',
        'Credit history',
        'Purpose',
        'Credit amount',
        'Savings account/bonds',
        'Present employment since', 
        'Installment rate in percentage of disposable income', 
        'Personal status and sex',
        'Other debtors / guarantors',
        'Present residence since',
        'Property',
        'Age in years', 
        'Other installment plans ',
        'Housing',
        'Number of existing credits at this bank',
        'Job',
        'Number of people being liable to provide maintenance for',
        'Telephone',
        'foreign worker',
        'Class',
    ]
    data.columns = col
    enc = OneHotEncoder(handle_unknown='ignore')
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    X_cat = data.select_dtypes(include=['object'])
    enc.fit(X_cat)
    names = list()
    for i in enc.categories_:
        names.extend(i)
    X_transform = pd.DataFrame(enc.transform(X_cat).toarray(), columns = names)
    data_result = pd.concat([data.select_dtypes(exclude=['object']), X_transform], axis=1)


    scaler = MinMaxScaler()
    scaler.fit(data.select_dtypes(exclude=['object']))
    X_new = scaler.transform(data.select_dtypes(exclude=['object']))
    X_ = np.concatenate([X_new, X_transform.to_numpy()], axis=1)
    
    return X_, y


if __name__ == "__main__":

    X_, y = data_processing('german.data')
    _cost_ = 1.0
    #num_qubits = 8 #8
    #dev = qml.device('default.qubit.torch', wires = num_qubits, shots=10000) # lightning.qubit
    #theta = np.array(list(range(4*num_qubits)))/(2*num_qubits)
    
    weights_init = 0.01 * np.random.randn(4*num_qubits, requires_grad=True)
    bias_init = np.array(0.0, requires_grad=True)

    opt = qml.SPSAOptimizer(250)
    #opt = qml.NesterovMomentumOptimizer(0.5)

    # looking at using QNG optimizer
    # looking at QNSPSA optimizer 
    
    weights = weights_init
    bias = bias_init
    #X_ = np.concatenate([X_new, X_transform.to_numpy()], axis=1) #data_res.to_numpy() X_new
    Y_ = y
    Y_ = Y_ * 2 - 3

    depth = 1
    batch_size = 5*depth #5
    cost_saved = []
    
    print(f'\nStarting variational process with depth {depth}\n')
    for it in range(100):
    
        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, len(X_), (batch_size,))
        X_batch = X_[batch_index]
        Y_batch = Y_[batch_index]
        
        weights, _, _, bias,_,_ = opt.step(cost2, weights, num_qubits, depth, bias, X_batch, Y_batch)
        #weights, _, _, bias,_,_ = opt.step_and_cost(circuit, X_batch, weights, num_qubits, 1)
        
        #params, loss = opt.step(cost2, weights, num_qubits, 1, bias, X_batch, Y_batch)
        #print(np.sign(variational_classifier(weights, X_batch[0], num_qubits, depth, bias)))
        # Compute accuracy
        
        predictions = [np.sign(variational_classifier(weights, x, num_qubits, depth, bias)) for x in X_]
        #print(list(zip(Y_ , predictions)))
        acc = accuracy(Y_, predictions)
        cost_ = cost2(weights, num_qubits, depth, bias, X_, Y_)
        cost_saved.append(cost_)
        
        
        logger.debug("Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} ".format(
                it + 1, cost_saved[-1], acc
        ))
        
        
        if _cost_ > cost_: 

            logger.debug(f'The 10 highest probabilities are for states {sorted(ansatz_2(weights, num_qubits, depth).items(), key=lambda x: x[1])[-10:]}')
            
            _cost_ = cost_ 

    
    fig = plt.figure(figsize=(16,10))
    plt.plot(cost_saved, "b", label="SPSA")
    #plt.plot(cost_saved_nes_mix, "r", label="SPSA mixed")
    #plt.plot(cost_saved_nes, "g", label="Nesterov")
    
    #plt.plot(qng_cost, "g", label="Quantum natural gradient descent")
    
    plt.ylabel("Cost function value")
    plt.xlabel("Optimization steps")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'SPSA_torch_depth_{depth}_2.png', format='png')
