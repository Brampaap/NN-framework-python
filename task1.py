import numpy as np #Matrix operation lib. Otherwise we should use too many loops.
import matplotlib.pyplot as plt
import not_torch as NN
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) # For pretty output

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
    # Dataset
    X = np.array([[1,1,0,1,1,1,1],
                  [0,0,0,1,0,1,0],
                  [0,1,1,1,1,0,1],
                  [0,1,1,1,0,1,1],
                  [1,0,1,1,0,1,0],
                  [1,1,1,0,0,1,1],
                  [1,1,1,0,1,1,1],
                  [0,1,0,1,0,1,0],
                  [1,1,1,1,1,1,1],
                  [1,1,1,1,0,1,0],
                 ])

    y = np.array([[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],
                  [0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],
                 ])
                 
    # Model init
    model = NN.layers.Sequential(
        NN.layers.Linear(7, 9),
        NN.layers.Sigmoid(),
        NN.layers.Linear(9, 9),
        NN.layers.Sigmoid(),
        NN.layers.Linear(9, 4),
        NN.layers.Sigmoid(),
    )
    
    # Loss
    criterion = NN.criterions.MSE()
    # Optimizer
    optimizer = NN.optimizers.RMSprop()
    # Params
    epochs = 270
    batch_size = 10
    learning_rate = 10
    
    history = []
    # Switch model to train mode
    model.train()
    
    for i in range(epochs):
    
        for x, y_true in loader(X, y, batch_size):
        
            # forward - calculate all values until loss
            y_pred = model.forward(x)
            
            # loss - calculate loss
            loss = criterion.forward(y_pred, y_true)
            
            # Backward pass though loss
            grad = criterion.backward(y_pred, y_true)
            
            # Backward pass though model
            model.backward(x, grad)
            
            # Make optimizer step
            optimizer.update(params    = model.parameters(),
                             gradients = model.grad_parameters(),
                             lr        = learning_rate,
                             )
            # Update loss history
            history.append(loss)
            
        
    # Loss plot
    plt.title("Training loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.plot(history, 'b')
    plt.show()
    
    # Print last loss value
    print(history[-1], ": LOSS")
    
    # Switch model to inference mode
    model.eval()
    
    # Prediction
    y_pred = model.forward(X)
    # Accuracy
    print(np.round(y_pred)==y)


if __name__ == "__main__":
    main()
