"""
This will be a file I use for my own plotting, to keep it distinct from the
plotting code from spragnur (though I should probably delete that file ...).
"""

import numpy as np
import matplotlib.pyplot as plt


def get_moving_average(arr, r=0.3):
    avg_arr = [arr[0]]
    for (index,value) in enumerate(arr):
        if index == 0:
            continue
        mv_avg = avg_arr[index-1]*(1-r) + value*r
        avg_arr.append(mv_avg)
    return np.array(avg_arr)


def main():

    # Load the two datasets. Shapes should be (100,5).
    results_myway = np.loadtxt(open("breakout_11-28-21-07_.csv", "rb"), 
                               delimiter=",", skiprows=1)
    results_control = np.loadtxt(open("breakout_11-29-08-20_.csv", "rb"), 
                               delimiter=",", skiprows=1)
    
    # A bunch of settings to tweak
    f, arr = plt.subplots(1,2, figsize=(20, 10))
    lsize = 25
    f_title = 40
    f_axis = 35
    lw = 6
    
    # Get average game scores
    i=0
    arr[i].set_title("Scores", fontsize=f_title)
    arr[i].set_xlabel("Training Epochs", fontsize=f_axis)
    arr[i].set_ylabel("Score Per Episode", fontsize=f_axis)
    #arr[i].plot(results_myway[:,3], 'b--', lw=lw, label="My Way")
    #arr[i].plot(results_control[:,3], 'k--', lw=lw, label="Default DQN")
    arr[i].plot(get_moving_average(results_myway[:,3]), 'b-', lw=lw,
            label="My Way (M.Avg.)") 
    arr[i].plot(get_moving_average(results_control[:,3]), 'k-', lw=lw,
            label="Default DQN (M.Avg.)")
    arr[i].legend(loc="upper left", prop={'size':lsize})
    
    # Get average Q-values encountered
    i=1
    arr[i].set_title("Q-Values", fontsize=f_title)
    arr[i].set_xlabel("Training Epochs", fontsize=f_axis)
    arr[i].set_ylabel("Average Action Value", fontsize=f_axis)
    arr[i].plot(results_myway[:,4], color='b', lw=lw, label="My Way")
    arr[i].plot(results_control[:,4], color='k', lw=lw, label="Default DQN")
    arr[i].legend(loc="lower right", prop={'size':lsize})
    
    # Layout set, then save.
    plt.tight_layout()
    plt.savefig("fig_ancas_class_presentation.png")


if __name__ == "__main__":
    main()
