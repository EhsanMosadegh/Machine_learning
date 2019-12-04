import numpy as np
# import matplotlib.pyplot as plt

def softmax(activation_vector):
    ''' Return softmaxed vector of given vector '''
    return np.exp(activation_vector) / np.sum(np.exp(activation_vector))

def _2D(_1Dy):
    '''Return probability distribution of given value'''
    if _1Dy == 0:
        return np.array([1,0])
    else: # == 1 
        return np.array([0,1])

def _1D(_2Dy): 
    ''' return most likely class given the probabilities '''
    if len(_2Dy) > 1:
        ans = np.array([])
        for point in _2Dy:
            if point[0] <= point[1]: # if ~ [0,1]
                ans = np.append(ans, 1)
            else: # if ~ [1,0]
                ans = np.append(ans, 0)
        return ans
    else:
        if _2Dy[0] <= _2Dy[1]: # if ~ [0,1]
            return 1
        else: # if ~ [1,0]
            return 0

def predict(model, x, twoD=False):
    ''' Helper function to predict an output (0 or 1)
    - model is the current version of the model:
    - {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2} 
    - It's a dictionary
    - x is one sample (without the label) '''

    #hidden layer matrix [num_features x num_hidden_nodes]
    hidden_weights_matrix = model['W1']
    #hidden layer bias vector length [num_hidden_nodes]
    hidden_bias_vector = model['b1']

    # calculate activation by matrix multiplying the feature matrix
    # by the weights matrix then add the bias vector

    # [num_features] @ [num_features x num_hidden_nodes] = [num_hidden_nodes]
    # [num_hidden_nodes] + [num_hidden_nodes]
    # activation vector is [num_hidden_nodes]
    hidden_activation_vector  = (x @ hidden_weights_matrix) + hidden_bias_vector 
    hidden_nonlinearity = np.tanh(hidden_activation_vector)

    # output layer weights [num_hidden_nodes x num_features]
    output_weights_matrix = model['W2']
    # output layer bias [num_features]
    output_bias_vector = model['b2']

    # [num_hidden_nodes] @ [num_hidden_nodes x num_features] 
    # = [num_features] + [num_features]
    output_activation_vector  = (hidden_nonlinearity @ output_weights_matrix) \
                                + output_bias_vector
    
    prediction = softmax(output_activation_vector)

    if twoD == True:
        return prediction
    else:
        return _1D(prediction)

def calculate_loss(model, X, y):
    ''' Helper function to evaluate the total loss on the dataset
    - Model is the current version of the model:
    - {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2} 
    - It's a dictionary
    - X is all the training data
    - y is the training labels '''
    loss = 0
    N = len(y)

    for training_sample, label in zip(X, y):
        y_hat = predict(model, training_sample, twoD=True)
        y_test = _2D(label)
        # sum of y * log(y_hat)
        loss += np.dot(y_test, np.log(y_hat))

    # multiply the sum by -(1/N)
    loss *= -(1/N)
    
    return loss

def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    ''' Learns parameters for the neural network and returns the model
    - X is the training data
    - y is the training labels
    - nn_hdim is the number of nodes in the hidden layer
    - num_passes is the number of passes through the training data for
    gradient descent 
    - print_loss: if true, print the loss every 1000 iterations '''

    eta = 0.01
    num_features = len(X[0])

    W1 = np.random.rand(num_features, nn_hdim)
    b1 = np.random.rand(nn_hdim)
    W2 = np.random.rand(nn_hdim, num_features)
    b2 = np.random.rand(num_features)

    current_model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2} 
    prev_loss = 1
    prev_prev_loss = 2
    for epoch in range(num_passes):
        for sample, label in zip(X, y):
            # yhat is [num_features]
            yhat = predict(current_model, sample, twoD=True)
            if label == 0:
                y_test = np.array([1, 0])
            else:
                y_test = np.array([0, 1])
            # thus dLdyhat is [num_features]
            dLdyhat = yhat - y_test

            # sample is [num_features]
            # W1 is [num_features x num_hidden_nodes]
            # thus sample @ W1 is [num_hidden_nodes]
            # b1 is [num_hidden_nodes]
            # thus activation and nonlinearity are [num_hidden_nodes]
            activation = sample @ W1 + b1
            nonlinearity = np.tanh(activation)

            # hT is [num_hidden_nodes x 1]
            # dLdyhat is [1 x num_features]
            # thus hT @ dLdyhat is [num_hidden_nodes x num_features] 
            # which is W2
            dLdW2 = np.transpose([nonlinearity]) @ [dLdyhat]
            
            # dLdyhat is [1 x num_features]
            # which is b2
            dLdb2 = dLdyhat
            
        
            # dLdyhat is [num_features]
            # W2T is [num_features x num_hidden_nodes]
            W2T = np.transpose(W2)
            # dLdyhat @ W2T is [num_hidden_nodes]
            dLdyhatW2T = dLdyhat @ W2T

            # fh is [num_hidden_nodes]
            fh = 1 - np.power(nonlinearity, 2)
            # dLdyhatW2T is [num_hidden_nodes]
            # element wise multiplication
            # thus dLda is [num_hidden_nodes]
            dLda = fh * dLdyhatW2T


            # xT is [num_features x 1]
            # dLda is [1 x num_hidden_nodes]
            # xT @ dLda is [num_features x num_hidden_nodes]
            # which is W1
            dLdW1 = np.transpose([sample]) @ [dLda]

            # dLda is [num_hidden_nodes]
            # which is b1
            dLdb1 = dLda 

            # update weights and bias
            W1 -= eta * dLdW1
            b1 -= eta * dLdb1
            W2 -= eta * dLdW2
            b2 -= eta * dLdb2

            current_model['W1'] = W1
            current_model['b1'] = b1
            current_model['W2'] = W2
            current_model['b2'] = b2

        if epoch % 1000 == 0: 
            this_loss = calculate_loss(current_model, X, y)
            if print_loss == True:
                print("# of Hidden Dimensions: ", nn_hdim, "  Epoch #: " , epoch, "   Loss: ", this_loss)
            if np.allclose(this_loss, prev_prev_loss) and np.allclose(this_loss, prev_loss):
                return current_model
            else:
                prev_prev_loss = prev_loss
                prev_loss = this_loss

    return current_model