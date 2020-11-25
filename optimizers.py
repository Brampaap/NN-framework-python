import numpy as np #Matrix operation lib. Otherwise we should use too many loops.
import matplotlib.pyplot as plt

class Optimizer():
    """An abstract class is purposed not to create all methods from scratch, but to reinitialize existing ones.
    Other layers inherit from this class.
    """
    def update(self):
        raise NotImplementedError

class SGD(Optimizer):
    def update(self, params, gradients, lr=1e-3):
        for weights, gradient in zip(params, gradients):
            weights -= lr * gradient
        return 0
    
class RMSprop(Optimizer):
    def __init__(self, decay=0.9):
        self.decay = decay
        self.EMSG = 0
        
    def update(self,params, gradients, lr=1e-3):
        if self.EMSG is 0:
            self.EMSG = np.ones_like(gradients)
        
        for weights, gradient, EMSG in zip(params, gradients, self.EMSG):
            EMSG = self.decay * EMSG + (1-self.decay) * gradient**2
            weights -= lr/(np.sqrt(EMSG)+ 1e-8) * gradient
        return 0

class Adam(Optimizer):
    """Adaptive Moment Estimation"""
    def __init__(self, beta1=0.9, beta2=0.999):
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = 0
        self.v = 0
        self.t = 1.00001
    
    def update(self,params, gradients, lr=1e-3):
        if (self.m and self.v) is 0:
            self.m = np.zeros_like(gradients)
            self.v = np.zeros_like(gradients)
        
        for weights, gradient, m, v in zip(params, gradients, self.m, self.v):
            m = self.beta1 * m + (1-self.beta1) * gradient
            v = self.beta2 * v + (1-self.beta2) * gradient**2
            
            m_hat = m/(1-self.beta1**self.t)
            v_hat = v/(1-self.beta2**self.t)
            
            weights -= lr/(np.sqrt(v_hat)+ 1e-8) * m_hat
        return 0
        
    
            
def main():
    pass

if __name__ == "__main__":
    main()
