import numpy as np #Matrix operation lib. Otherwise we should use too many loops.
import matplotlib.pyplot as plt
import not_torch as NN

def loader(X, Y, batch_size):
    """
    Data loader.
    This generator shuffles all indexes and takes `batch_size` samples.
    @ X - features dataset
    @ Y - answers dataset
    
    @ yield `X[batch_idx], Y[batch_idx]`
    """
    n = X.shape[0]
    
    # Shuffle all indexes at the start of each epoch
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    for start in range(0, n, batch_size):
        # Take not full batch at the last epoch is possible
        end = min(start + batch_size, n)
        
        batch_idx = indices[start:end]
    
        yield X[batch_idx], Y[batch_idx]

def main():
    pass

if __name__ == "__main__":
    main()
