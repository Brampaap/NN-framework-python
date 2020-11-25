import numpy as np #Matrix operation lib. Otherwise we should use too many loops.
import matplotlib.pyplot as plt

"""CRITERIONS"""

class Criterion():
    """An abstract class is purposed not to create all methods from scratch, but to reinitialize existing ones.
    Other layers inherit from this class.
    """
    def forward(self, input, target):
        raise NotImplementedError

    def backward(self, input, target):
        raise NotImplementedError
        
class MSE(Criterion):
    """
    Mean Squard Error loss
    """
    def forward(self, input, target):
        self.batch_size = input.shape[0]
        self.output = np.sum(np.power(target - input , 2)) / self.batch_size
        return self.output
 
    def backward(self, input, target):
        self.grad_output  = (2/self.batch_size *(input - target) )
        return self.grad_output

class CrossEntropy(Criterion):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

        
    def forward(self, input, target):
        input_clamp = np.clip(input, self.eps, 1 - self.eps)
        
        self.N = target.shape[0]

        self.output = np.sum( (target==1) * -np.log(input_clamp) - ( (target==0) * np.log(1-input_clamp) ) ) / self.N

        return self.output

    def backward(self, input, target):
 
        input_clamp = np.clip(input, self.eps, 1 - self.eps)
        
        grad_input = -(target-input_clamp) * input_clamp / self.N

        return grad_input

def main():
    pass

if __name__ == "__main__":
    main()
