# Multi Armed Bandit from Scratch

This is a project which was inspired form the 2020 Christmas Kaggle Competition where two teams of elves have to collect as many candy canes as possible from vending machines with different rewards probabilities. The team with the most candy win.

This project is has been created as a 'code-along' for anyone who wants to gain a basic understanding on multi armed bandit and how it works.

## Prerequisites and Imports
The first step is to import the necessary libraries. These are commonly used ones so the experiement is reproducible.
```python: 
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```
We then create a function that will help us visualise the history of the agent's interactions with the environment.
```python: 
def plot_history(history):
    rewards = history['rewards']
    avg_rewards = history['avg_rewards']
    chosen_machines = history['machines']
    
    fig = plt.figure(figsize=[20,6])
    
    line = fig.add_subplot(1,2,1)
    line.plot(avg_rewards, color='C3', label='avg rewards')
    line.set_title('Average Rewards')
    
    bchart = fig.add_subplot(1,2,2)
    bchart.bar([i for i in range(len(chosen_machines))], chosen_machines, color='C3', label='chosen machines')
    bchart.set_title('Chosen Actions')
```

## Create the environment
First we need to create an environment. We will set the probability of payout and how much the environment actually pays out.
