import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_loss_acc(train_loss, val_loss,fig_name):
    x = np.arange(len(train_loss))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    lns1 = ax1.plot(x, train_loss, 'y-', label='train_loss')
    lns2 = ax1.plot(x, val_loss, 'g-', label='val_loss')
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    fig.tight_layout()
    plt.title(fig_name)

    plt.savefig(os.path.join('./diagram', fig_name))

