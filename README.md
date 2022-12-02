# Introduction to Multi Armed Bandit

![multi armed bandit](https://github.com/mriffaud/Multi-Armed-Bandit/blob/main/images/multi%20armed%20bandit.png)

This is a project which was inspired from the 2020 Christmas Kaggle Competition where two teams of elves have to collect as many candy canes as possible from vending machines with different rewards probabilities. The team with the most candy win.

This project is has been created as a 'code-along' for anyone who wants to gain a basic understanding on multi armed bandit and how it works.

## Prerequisites and Imports
The first step is to import the necessary libraries. These are commonly used ones so the experiment is reproducible.
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
First we need to create a class and environment. We will set the probability of payout and how much the environment actually pays out.
We start with creating a class called ```Env()``` and the ```__init__()``` constructor which is going to store our two arguments: the rewards probabilitites and the actual rewards.
The reason this technique is called a multi armed bandit is because the agent has multiple options to chose from so in our implementation of the environment we also want to know how many machines are available. Therefore, our next step is to check that the length of our arguments ```rewards_probas``` and ```rewards``` are equal, if not, the the environment will be invalid and the function will return an error message.
We then pass the arguments ```rewards_probas``` and ```rewards``` to the environment and the number of machines that are available: ```k_machines```
        
Now that we have our environment ready we can build an agent to interact with it so we are going to define a function called ```choose_machine()```. In this function, we create an ```if statement``` to return an error message if any machine is less than 0 or greater than the total number of machines as this would be outside our specified environment. If no exception is raised, the function returns the ```rewards``` for that machine. In this example each machine is internally guided by a reward rate so we generate any ramdom number and if that ramdom number is less than the reward probability of that particular machine then it gives the reward as stated in the reward list otherwise it gives you zero.

```python:
# create a class for the environment that contains the probability to get a reward and the actual rewards
class Env(object):
    def __init__(self, prob_reward, rewards):
        if len(prob_reward) != len(rewards): 
            raise Exception(f'size of reward probability: {len(prob_reward)} does not does match size of rewards: {len(rewards)}')
        
        # pass arguments to the environment
        self.prob_reward = prob_reward
        self.rewards = rewards
        self.k_machines = len(rewards)
        
    # define function to specify the machine the elf is going to use 
    def choose_machine(self, machine):
        if machine < 0 or machine > self.k_machines:
            raise Exception(f'machine must be a value between 0 and {self.k_machines -1}') # -1 because 0 counts as a number
        return self.rewards[machine] if np.random.random() < self.prob_reward[machine] else 0.0
```

Now that we have our class we can create the environment which is going to make it more understandable. We create an environment with the rewards probabilities and the rewards:
```python:
environment = Env(prob_reward=[0.01, 0.05, 0.20, 0.50, 0.65, 0.90], 
                  rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
```
This simply means that the first machine has a probability of 0.01% chance to give you 1, in our example this would be 1 candy cane.

The objectif of a multi armed bandit is that at the end of the day the agent will be able to make the wisest decision and choose the machine that has the highest probability of payout. In this example it is very easy for a human to determine that the best machine to use in the machine number 6 as it has a 90% of rewarding us.

Since we have created the environment where we stated the rewards probabilities we can see that the best machine to use is the machine number 6 but in reality we do not know the rewards probabilities therefore it is more challenging to know which machine to chose from and how often. This is the reason why building an agent is a great tool to automatically learn the pay rate for each machine and also take optimal actions.

Now to the fun part! Since we have our class and environment the next step is to build our agent, in this context it is an elf but we want to build different variants of agents/elves with different level of intelligence to illustrate the multi armed bandit problem.

So first we are going to build a base line agent that will enable us to compare its performance with our intelligent ones.

## Create a base line agent
To have a base line to compare our 'intelligent' agents against we need to create random agent. We start by creating a class called ```RandomAgent()``` which is going to interact with the environment we created above.
This time, the the ```__init__()``` constructor takes the environment but we also specify the ```max_iterations``` which is the maximum number of steps the agent can take. In this example we set as the default ```max_iterations=2000``` so in the 2000 steps the goal is for the agent to make the best cumulative reward possible.

The next step is to define the ```action()``` function which let our agent take action with the environemnet we created. We do not expect this agent to optimise its cumulative rewards and take the widest decisions since it should behave randomly.

In order to be able to visualise what is happening behind the scene, we want to track of the rewards the agent generates and which machine it has used: 
* we start by creating a list ```machine_counts``` that stores which machine the agent use for each iteration and initialise it with zero
* we also want the actual reward the agent is generating, so we are creating a list that stores all the rewards at each step. We call it ```rewards```
* finally, we want the average of reward generated by the agent over time and store it in ```avg_rewards```

This agent isn't an intelligent one, so it is going to make a random choice out of the number of machine available, next we want to see how much reward the agent generates each time using the ```choose_machine()``` function that contains the rewards based on the values we supplied to the environment. 
Finally, we want to increase the ```machine_counts``` to know how many time that particular machine has been selected by the agent.

At the end of our ```RandomAgent()```, we create a dictionary with the results that we will be able to use in our plotting function to see and comapre the performance of our agents.

```python:
# create random agent
class RandomAgent(object):
    def __init__(self, env, max_iterations=2000):
        self.env = env 
        self.iterations = max_iterations 
        
    # let the agent take actions in the environment
    def action(self):
        # keep track of the rewards the agent generates and which machine it is using
        machine_counts = np.zeros(self.env.k_machines)
        
        # store the reward the agent is generating
        rewards= []
        # store reward the agent is generating over time
        avg_rewards = []
        
        for i in range(1, self.iterations+1):
            machine = np.random.choice(self.env.k_machines)
            reward = self.env.choose_machine(machine)
            
            # increasing the machine count to know how many time the machine has been used
            machine_counts[machine] += 1 
            
            # append the results to the list rewards and avg_rewards
            rewards.append(reward)
            avg_rewards.append(sum(rewards)/len(rewards))
            
        # create a dictionary with the results to use our plotting function
        return {'machines': machine_counts,
                'rewards': rewards,
                'avg_rewards': avg_rewards
               }
```
Now that we have our random agent ready, we can create an instance and see how the agent behaves.

```python:
# create the instance
random_agent = RandomAgent(env=environment, max_iterations=2000)

# action the agent is taking
ra_history= random_agent.action()

# print
print(f'total reward : {sum(ra_history["rewards"])}')

total reward : 769.0
```
We can see that after a total of 2000 iterations, the agent made a total reward of 769 candy canes. But we can't tell which machine has been selected unless we plot it. This is the reason why we created the ```plot_history()``` function which takes the dictionary that the ```action()``` function returns and plots the average reward and how many times each machine was used.

```python:
# plot the history
plot_history(ra_history)
```
![random agent](https://github.com/mriffaud/Multi-Armed-Bandit/blob/main/images/random%20agent.png)

The barchart shows the agent has been using the machines randomly and the average cumulative reward is not increasing over time.

So now that we have our base case scenario with our random agent, let's build our 'intelligent' agents.

## Create 'intelligent' agents
### Epsilon Greedy Agent
The first intelligent agent we are going to build is called epsilon greedy. This agent explores randomly the different variations/machines available and keep track of how much each machine is rewarding but because there is a limited amount of time or iterations, there is a point at which the agent has to stop exploring and start exploiting what it has learnt from the environment. Exploitation means taking the best possible actions based on the information you have available at the time while exploration means investigating the options available.

What the epsilon greedy algorithm helps us to do is to solve one of the common situation in reinforcement learning which is called the exploration/exploitation dilemma where you need to keep a balance between exploring your environment and exploiting it.

The first step is to create a class called ```EpsilonGreedyAgent()```. In the initialiser, we will have the same as in our random agent, namely the constructor, environment, and maximum iterations but also epsilon. The way this parameter works is that for epsilon with epsilon probability it explores and with 1 minus epsilon probability it exploits. In this example it means that out of a hundred steps or iterations, the agent is going to take one random action instead of the wisest action it knows so far. This is important because the best known action the agent is aware of at that time may not be the actual widest one therefore if it explores more it can identify another machine that is better than the one it knows at that step. However, because we have a limited amount of time in the environment, we do not want to spend too long exploring otherwise we will consume all of the available time steps and won't cumulate the greatest amount of rewards.
        
As we said previously, our intelligent agent has to keep track of the rewards when exploring the environment, the way we do that is by using the Q values which gives you the pay rate for each of the machine. In other words, it the Q value provides the rewards per iteration.
Similarly, as for the random agent we create a list that stores all the rewards and cumulative average reward.

Once we have all our trackers sorted, we can use the epsilon probability inside a for loop by generating a random number and if that random number is less than epsilon, we want to explore otherwise we want to take the action that has the maximum reward stored in ```q_values``` where ```argmax()``` returns the index of the Q values where the it is the highest.

From there the rest of the codes are similar to our random agent where we store the reward the agent generates each step, we increase the total we have obtained by pulling the machine using ```reward```, we add 1 to ```machine_counts``` to indicate that the agent has chosen this machine x number of times. then, we use ```machine_rewards``` and the ```machine_counts``` to evaluate the Q values for that particular machine: ```q_values``` is equal to the total reward we got by using that machine divided by the number of times we have used the machine, this gives us the pay rate for that machine. It is an estimation of how valuable that particular machine is to our agent.
            
Finally, we retain the same dictionary as for the random agent to plot the performance of our agent.

```python:
# create epsilon greedy agent
class EpsilonGreedyAgent(object):
    def __init__(self, env, max_iterations=2000, epsilon=0.01):
        self.env = env 
        self.iterations = max_iterations
        self.epsilon = epsilon
        
    # let the agent take actions in the environment
    def action(self):
        # initialise the q_values
        q_values = np.zeros(self.env.k_machines)
        # keep track of total reward generated by each machine
        machine_rewards = np.zeros(self.env.k_machines)
        # keep track of the rewards the agent generates and which machine it is using
        machine_counts = np.zeros(self.env.k_machines)
        
        # store the reward the agent is generating
        rewards= []
        # store reward the agent is generating over time
        avg_rewards = []
        
        for i in range(1, self.iterations+1):
            machine = np.random.choice(self.env.k_machines) if np.random.random() < self.epsilon else np.argmax(q_values)
            
            # store information
            reward = self.env.choose_machine(machine)
            machine_rewards[machine] += reward 
            machine_counts[machine] += 1
            q_values[machine] = machine_rewards[machine]/machine_counts[machine]
        
            # append the results to the list rewards and avg_rewards
            rewards.append(reward)
            avg_rewards.append(sum(rewards)/len(rewards))
            
        # create a dictionary with the results to use our plotting function
        return {'machines': machine_counts,
                'rewards': rewards,
                'avg_rewards': avg_rewards
               }
```
Now we create an instance for the epsilon greedy agent to see its behaviour:

```python:
# create an instance of the epsilon greedy agent
egreedy_agent = EpsilonGreedyAgent(env=environment, max_iterations=2000, epsilon=0.1)
eg_history = egreedy_agent.action()

# print
print(f'total reward : {sum(eg_history["rewards"])}')

total reward : 1639.0
```

```python:
#let's call plot_history to see what actually happened
plot_history(eg_history)
```
![epsilon greedy agent](https://github.com/mriffaud/Introduction-to-Multi-Armed-Bandit/blob/main/images/epsilon%20greedy%20agent.png)

We can see that our epsilon greedy agent has generated a much larger amount of rewards than the random agent. We can also see that the average reward for the epsilon greedy agent is constantly improving until reaching the optimal action and hovering around 0.8 average reward per action. From the plot of the machine count we notice that the agent has settled to make the best of the machine 5 and used it a lot more than any other machines available. Looking back at the rewards probabilities, this is because the machine number 56 has 90% chance to reward our agent with a candy cane.

### Epsilon Greedy Agent with decay
Another variant of the epsilon greedy is epsilon greedy with decay which reduces the probability of exploration over time because once you have learnt enough action values you don't need to keep exploring. Thus, there is a point at which the agent should have explored its environment enough and knows the best action to choose from and therefore the exploration rate should reduce. Otherwise, the agent waste time (iterations) that should be used to make an informed decision making a random one which ultimately is going to limit the total amount of final reward.

So using exactly the same codes as for the epsilon greedy agent above, we introduce the decay constant in our class ```EpsilonGreedyDecayAgent()```. We create our initialiser and add the ```decay``` constant: it should be a number between 0 and 1 and a ```decay_interval``` so after every decay interval time steps this will decay ```epsilon```.
After we evaluate everything in the for loop, we create an if statement for the decay of epsilon. For this, we use the mod (or modulo) which is the is the remainder of a division, thus if the iteration mod the decay interval is zero than it has gone through a decay interval time step therefore we multiply ```epsilon``` by a ```decay_interval``` which reduce ```epsilon``` and consequently the exploration rate.
        
Finally  we retain the same dictionary to plot the agent's performance.

```python:
# create epsilon greedy agent with decay
class EpsilonGreedyDecayAgent(object):
    def __init__(self, env, max_iterations=2000, epsilon=0.01, decay=0.001, decay_interval=50):
        self.env = env 
        self.iterations = max_iterations
        self.epsilon = epsilon
        self.decay = decay
        self.decay_interval = decay_interval
        
    # let the agent take actions in the environment
    def action(self):
        # initialise the q_values
        q_values = np.zeros(self.env.k_machines)
        # keep track of total reward generated by each machine
        machine_rewards = np.zeros(self.env.k_machines)
        # keep track of the rewards the agent generates and which machine it is using
        machine_counts = np.zeros(self.env.k_machines)
        
        # store the reward the agent is generating
        rewards= []
        # store reward the agent is generating over time
        avg_rewards = []
        
        for i in range(1, self.iterations+1):
            machine = np.random.choice(self.env.k_machines) if np.random.random() < self.epsilon else np.argmax(q_values)
            
            # store information
            reward = self.env.choose_machine(machine)
            machine_rewards[machine] += reward 
            machine_counts[machine] += 1
            q_values[machine] = machine_rewards[machine]/machine_counts[machine]
        
            # append the results to the list rewards and avg_rewards
            rewards.append(reward)
            avg_rewards.append(sum(rewards)/len(rewards))
            
            # if statement for the decay of epsilon at interval step time
            if i%self.decay_interval == 0:
              self.epsilon = self.epsilon * self.decay
              
        # create a dictionary with the results to use our plotting function
        return {'machines': machine_counts,
                'rewards': rewards,
                'avg_rewards': avg_rewards
               }
```
We can now create an instance for the ```EpsilonGreedyDecayAgent()``` to look at its behaviour:

```python:
# create an instance of the epsilon greedy agent with decay
egreedyd_agent = EpsilonGreedyDecayAgent(env=environment, max_iterations=2000, epsilon=0.01, decay=0.001, decay_interval= 50)
egd_history = egreedyd_agent.action()

# print total rewards and charts
print(f'total reward : {sum(egd_history["rewards"])}')

total reward : 1781.0
```

```python:
#let's call plot_history to see what actually happened
plot_history(egd_history)
```
![epsilon greedy agent with decay](https://github.com/mriffaud/Introduction-to-Multi-Armed-Bandit/blob/main/images/epsilon%20greedy%20agent%20with%20decay.png)

We can see that our epsilon greedy agent with decay has generated a greater amount of rewards than the two previous agents. While the epsilon greedy agent was constantly improving its average reward over the 2000 iterations, we notice that in the case of the epsilon greedy agent with decay reducing the exploration time has significantly improve the learning time as by the 100th step the agent had already learnt that the machine number 5 was the one providing the best reward so the agent was able to use that knowledge without wasting iterations making random actions and therefore maximising its reward. Thus ```epsilon``` and ```decay_interval``` become the hyperparameters that we would need to tune depending on the problem we are trying to solve.

Please note that if you re-run the ```EpsilonGreedyDecayAgent()``` without making any amendment to the codes, the results will keep changing. This is because there is a part of randomness in our models.
