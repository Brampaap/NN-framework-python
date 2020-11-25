import numpy as np #Matrix operation lib. Otherwise we should use too many loops.
import matplotlib.pyplot as plt
import my_lib as NN
from tkinter import *
import tkinter
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

    X = np.array([  [ 1, 1, 1, 1, 1, 1, 0 ],#0
                    [ 0, 1, 1, 1, 1, 0, 0 ],#1
                    [ 0, 1, 1, 0, 1, 1, 1 ],#2
                    [ 0, 1, 1, 1, 1, 0, 1 ],#3
                    [ 1, 0, 1, 1, 0, 0, 1 ],#4
                    [ 1, 1, 0, 1, 1, 0, 1 ],#5
                    [ 1, 1, 0, 1, 1, 1, 1 ],#6
                    [ 0, 1, 1, 1, 0, 0, 0 ],#7
                    [ 1, 1, 1, 1, 1, 1, 1 ],#8
                    [ 1, 1, 1, 1, 0, 0, 1 ]#9
                    ])
    y = np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]])

    model = NN.layers.Sequential(
        NN.layers.Linear(7, 9),
        NN.layers.LeakyReLU(),
        NN.layers.Linear(9, 9),
        NN.layers.LeakyReLU(),
        NN.layers.Linear(9, 1),
        
        # Архитектура  inp(7) x 9 x 9 x out(1)

    )

    criterion = NN.criterions.MSE()

    epochs = 2000
    batch_size = 10
    learning_rate = 1e-2
    optimizer = NN.optimizers.RMSprop()

    history = []
    model.train()
    for i in range(epochs):
        for x, y_true in loader(X, y, batch_size):
            # forward -- считаем все значения до функции потерь
            y_pred = model.forward(x)
            loss = criterion.forward(y_pred, y_true)
            
            # backward -- считаем все градиенты в обратном порядке
            grad = criterion.backward(y_pred, y_true)
            model.backward(x, grad)
            # обновляем веса
            optimizer.update(model.parameters(),
                model.grad_parameters(),
                learning_rate)
            
            
            history.append(loss)
            

    print(history[-1], ": LOSS")
    model.eval()
    
    root = Tk()
    root.title("GUI")
    root.geometry("200x200")

    def start():

        number = [var1.get(), var2.get(), var3.get(),
                  var4.get(), var5.get(), var6.get(), var7.get()]
        model.eval()

        ans.set(str(np.round(model.forward(number))))


    frame = Frame()
    frame.pack(side=LEFT)
    var1 = IntVar()
    c1 = Checkbutton(frame, variable=var1, onvalue=1, offvalue=0, padx=1, pady=1)
    c1.grid(row=1, column=0, sticky=W)
    var2 = IntVar()
    c2 = Checkbutton(frame, variable=var2, onvalue=1, offvalue=0, padx=1, pady=1)
    c2.grid(row=0, column=1, sticky=W)
    var3 = IntVar()
    c3 = Checkbutton(frame, variable=var3, onvalue=1, offvalue=0, padx=1, pady=1)
    c3.grid(row=1, column=2, sticky=W)
    var4 = IntVar()
    c4 = Checkbutton(frame, variable=var4, onvalue=1, offvalue=0, padx=1, pady=1)
    c4.grid(row=3, column=2, sticky=W)
    var5 = IntVar()
    c5 = Checkbutton(frame, variable=var5, onvalue=1, offvalue=0, padx=1, pady=1)
    c5.grid(row=4, column=1, sticky=W)
    var6 = IntVar()
    c6 = Checkbutton(frame, variable=var6, onvalue=1, offvalue=0, padx=1, pady=1)
    c6.grid(row=3, column=0, sticky=W)
    var7 = IntVar()
    c7 = Checkbutton(frame, variable=var7, onvalue=1, offvalue=0, padx=1, pady=1)
    c7.grid(row=2, column=1, sticky=W)
    rightframe = Frame().pack(side=RIGHT)
    ans = StringVar()

    e = Entry(rightframe, textvariable=ans)
    e.pack()
    button = Button(text="Распознать",
                command=start).pack()

    root.mainloop()
    
    


if __name__ == "__main__":
    main()
