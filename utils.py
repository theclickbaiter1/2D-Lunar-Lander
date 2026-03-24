import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(scores, x=None, window=100, filename='learning_curve.png'):
    """
    Plots the learning curve with a moving average.
    
    Args:
        scores (list): The list of scores per episode.
        x (list): Optional list of episode numbers.
        window (int): The window size for the moving average.
        filename (str): The filename to save the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if x is None:
        x = [i+1 for i in range(len(scores))]
        
    ax.plot(x, scores, label='Score', alpha=0.5)
    
    # Calculate moving average
    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        # Adjust x-axis for moving average
        x_ma = np.arange(window, len(scores) + 1)
        ax.plot(x_ma, moving_avg, color='red', label=f'{window}-Episode Moving Average')
        
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('DQN Training Agent')
    plt.legend()
    plt.savefig(filename)
    plt.close(fig)
