import numpy as np #Matrix operation lib. Otherwise we should use too many loops.
import matplotlib.pyplot as plt

class Module():
    """
    An abstract class is purposed not to create all methods from scratch, but to reinitialize existing ones.
    Other layers inherit from this class.
    """
    def __init__(self):
        self._train = True

    def forward(self, input):
        raise NotImplementedError

    def backward(self,input, grad_output):
        raise NotImplementedError

    def parameters(self):
        """
        Returns its parameters (bias, weights), which are trained as NumPy array
        """
        return []

    def grad_parameters(self):
        """
        Returns a NumPy list of gradient tensors for its parameters
        """
        return []
    
    # Some layers differ in behavior in learning and inference modes
    def train(self):
        self._train = True

    def eval(self):
        self._train = False

class Sequential(Module):
    def __init__ (self, *layers):
        super().__init__()
        self.layers = layers
    
    def forward(self, input):
        """
        Forward pass though all layers:
            
            y[0] = layers[0].forward(input)
            y[1] = layers[1].forward(y[0])
            ...
            output = module[n-1].forward(y[n-2])
        
        It should be a simple for loop: for a layer in layers ...
        
        No need to store outputs again:
        """
        
        for layer in self.layers:
            input = layer.forward(input)
        
        self.output = input
        
        return self.output
        
    def backward(self, input, grad_output):
        """
        Backward pass of backprop algorithm.
        
        Backward designed for:
            1. Calculation of gradients for own parameters
            2. Pass a gradient relative to its input
            
        Each module calculates its own gradients.
        All that is needed is to implement the transfer of the gradient.
        """
        for i in range(len(self.layers)-1, 0, -1):
            grad_output = self.layers[i].backward(self.layers[i-1].output, grad_output)
        
        grad_input = self.layers[0].backward(input, grad_output)
        
        return grad_input
        
    def parameters(self):
        """
        Get all trainable parameters for each layer
        """
        res = []
        for l in self.layers:
            res += l.parameters()
        return res
        
    def grad_parameters(self):
        """
        Get all gradients for each layer
        """
        res = []
        for l in self.layers:
            res += l.grad_parameters()
        return res
        
    """MODE SWITCHERS"""
     
    def train(self):
        for layer in self.layers:
            layer.train()
    
    def eval(self):
        for layer in self.layers:
            layer.eval()

# Layers
class Linear(Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        
        # Initializing weight parameters with ReLU Xavier
        stdv = 1./np.sqrt(dim_in)
        self.W = np.random.uniform(-stdv, stdv, size=(dim_in, dim_out))
        self.b = np.random.uniform(-stdv, stdv, size=(1,dim_out))
    
    def forward(self, input):
        self.output = np.dot(input, self.W) + self.b
        return self.output
    
    def backward(self, input, grad_output):
        self.grad_b = np.mean(grad_output, axis=0)
        
        #     in_dim x batch_size
        self.grad_W = np.dot(input.T, grad_output)
        #                 batch_size x out_dim

        grad_input = np.dot(grad_output, self.W.T)
        
        return grad_input
        
    def parameters(self):
        return [self.W, self.b]

    def grad_parameters(self):
        return [self.grad_W, self.grad_b]

class ReLU(Module):
    """
    ReLU - non linear activation function.
    
            | x, x >= 0
    ReLU = <|
            | 0, x < 0
    """
    def __init__(self):
         super().__init__()
    
    def forward(self, input):
        self.output = np.maximum(input, 0)
        return self.output
    
    def backward(self, input, grad_output):
        grad_input = np.multiply(grad_output, input >= 0)
        return grad_input

class LeakyReLU(Module):
    """
    Modified version of ReLU with useful derivative at x < 0
    
            | x, x >= 0
    ReLU = <|
            | slope * x, x < 0
            
    Where slope is a small constant.
    """
    def __init__(self, slope=0.01):
        super().__init__()
        self.slope = slope
        
    def forward(self, input):
        coefs = (input >= 0) + (input < 0) * self.slope
        self.output = np.multiply(input, coefs)
        return self.output
    
    def backward(self, input, grad_output):
        temp = np.where(input >= 0, 1, self.slope)
        grad_input = np.multiply(grad_output, temp)
        return grad_input

class Sigmoid(Module):
    """
    Sigmoid activation
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.output = 1/(1+np.exp(-input))
        return self.output
    
    def backward(self, input, grad_output):
        grad_input = (self.output * (1-self.output))*grad_output
        return grad_input

class SoftMax(Module):
    """
    Softmax activation function.
    """
    def __init__(self):
         super().__init__()
         
    def forward(self, input):
        # An important detail. If the inputs are large, then np.exp will be even larger
        # Stabilization
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        self.output = np.exp(self.output)
        self.output = self.output/np.sum(self.output, axis=1, keepdims=True)
        
        return self.output
    
    def backward(self, input, grad_output):
        # Diagonal case
        diagonal_els = np.diag(grad_output)
        
        diagonal_els = np.multiply(-diagonal_els,diagonal_els)
        np.fill_diagonal(grad_output,0)
        
        # Non-diagonal case
        grad_input = np.multiply(grad_output,1-grad_output)
        np.fill_diagonal(grad_input, diagonal_els[:,None])
        grad_input = np.multiply(input, grad_input)
        return grad_input

class Dropout(Module):
    """
    Dropout regularization.
    This layer randomly zeroes the input weights.
    Behaves differently during training and evaluation.
    """
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
        self.mask = None
        
    def forward(self, input):
        if self._train:
            self.mask = np.random.binomial(1,self.p, size=input.shape)
            self.output = self.mask*input
        else:
            self.output = input * self.p
        return self.output
    
    def backward(self, input, grad_output):
        if self._train:
            grad_input = grad_output * self.mask
        else:
            assert 1,"Something went wrong, you shouldn't use '.backward()' at the inference time"
        return grad_input
        
class BatchNorm(Module):
    """
    BatchNorm layer. (1D)
    The main idea of BatchNorm is this: for the current minibatch while training,
    in each hidden layer, we normalize the activations so that its distribution
    is Standard Normal (zero mean and one standard deviation).
    https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
    """
    def __init__(self, num_features, gamma=1, beta=0):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.num_features = num_features
        self.mu = np.ones((num_features,1))
        self.sigma = np.ones((num_features,1))
    
    def forward(self, input):
        if self._train:
            mu = np.mean(input, axis=0)
            sigma = np.var(input, axis=0)
            input_norm = (input - mu) / np.sqrt(sigma + 1e-05)
            
            self.output = self.gamma * input_norm + self.beta
            self.mu = 0.9 * self.mu + 0.1 * mu[:,None]
            self.sigma = 0.9 * self.sigma + 0.1 * sigma[:,None]
            
        else:
            input_norm = ((input - (self.mu.T * np.ones(input.shape))) / np.sqrt(self.sigma.T*np.ones(input.shape)) + 1e-05)
            self.output = self.gamma * input_norm + self.beta
        return self.output
    
    def backward(self, input, grad_output):
        if self._train:
            input_mu = input - np.mean(input, axis=0)

            std_inv = 1. / np.sqrt(np.var(input, axis=0) + 1e-05)
            
            dinput_norm = grad_output * self.gamma
            dsigma = np.sum(dinput_norm * input_mu, axis=0) * - 0.5 * std_inv**3
            dmu = np.sum(dinput_norm * -std_inv, axis=0) + dsigma * np.mean(-2.0 * input_mu, axis = 0)
            
            grad_input = (dinput_norm * std_inv) + (dsigma * 2 * input_mu / self.num_features) + (dmu / self.num_features)
            
        else:
            assert 1, "Switch to train()"
        return grad_input


if __name__ == "__main__":
    main()
