import numpy as np #Matrix operation lib. Otherwise we should use too many loops.
import matplotlib.pyplot as plt
import my_lib as NN
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

    model = NN.layers.Sequential(
        NN.layers.Linear(7, 9),
        NN.layers.Sigmoid(),
        NN.layers.Linear(9, 9),
        NN.layers.Sigmoid(),
        NN.layers.Linear(9, 4),
        NN.layers.Sigmoid(),
        
        # Архитектура  inp(7) x 9 x 9 x out(4)
    )

    criterion = NN.criterions.MSE()

    epochs = 270
    batch_size = 10
    learning_rate = 10
    optimizer = NN.optimizers.RMSprop()

    history = []
    model.train()
    for i in range(epochs):
        for x, y_true in loader(X, y, batch_size):
            # forward -- считаем все значения до функции потерь
            y_pred = model.forward(x)
            loss = criterion.forward(y_pred, y_true)
            #print(y_pred, y_true)
            #print('SUM OF SQUARES:', np.mean(np.power(y_pred-y_true, 2)))
            
            # backward -- считаем все градиенты в обратном порядке
            grad = criterion.backward(y_pred, y_true)
            model.backward(x, grad)
            # обновляем веса
            optimizer.update(model.parameters(),
                model.grad_parameters(),
                learning_rate)
            
            
            history.append(loss)
            
        

    plt.title("Training loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.plot(history, 'b')
    plt.show()

    print(history[-1], ": LOSS")
    model.eval()

    Y_pred = model.forward(X)
    print(np.round(Y_pred)==y) # Если всё True, значит каждое значение предсказано правильно.


if __name__ == "__main__":
    main()
