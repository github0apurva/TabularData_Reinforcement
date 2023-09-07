# How to train reinforcement learning model? with tabular data

**[RL_TRAINING](jupyter_notebook:RL_TRAINING)** 

## Basic setup before running the notebook
1. Check that your environment contains all the required packages installed

Source code:   _imbDRL_ : https://github.com/Denbergvanthijs/imbDRL
## 1. Import packages 

```
#built-in functions
from tensorflow.keras.layers import Dense, Dropout #define layers
import os
import pandas as pd

from RL_agent import TrainDDQN # RL training agents 
```

## 2. Define Hyperparameters
Before training the RL models, the hyperparameters need to be specified. Make sure to change the ```min_class``` and ```maj_class``` based on the labels of the dataset. 
```
episodes = 100_000  # Total number of episodes
warmup_steps = 170_000  #120,000 # Amount of warmup steps to collect data with random policy
memory_length = warmup_steps  # Max length of the Replay Memory
batch_size = 32 #64 
collect_steps_per_episode = 2000
collect_every = 500

target_update_period = 800  # Period to overwrite the target Q-network with the default Q-network
target_update_tau = 1  # Soften the target model update
n_step_update = 1

layers = [Dense(256, activation="relu"), Dropout(0.2),
          Dense(256, activation="relu"), Dropout(0.2),
          Dense(2, activation=None)]  # No activation, pure Q-values

learning_rate = 0.00025  # Learning rate
gamma = 0.7   # Discount factor #0.0 means only cares about rewards in the immediate state\
#gamma = 0.1
min_epsilon = 0.5  # Minimal and final chance of choosing random action
decay_episodes = episodes // 10  # Number of episodes to decay from 1.0 to `min_epsilon``

min_class = [1]  # Minority classes
maj_class = [0]  # Majority classes
```

## 3. Convert training, (optional)validation and test set to numpy 
Since model fitting requires input to be np.ndarray, convert the dataset to be trained and tested into numpy 
```
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()
X_val = X_valid.to_numpy()
y_val = y_valid.to_numpy()
```
## 4. Train, predict and evaluate 
``` class TrainDDQN()``` serves as wrapper for training, fitting, predicting and evaluating function. 
The input for the class are the defined hyperparameters: 
- episodes 
- warmup_steps 
- learning_rate
- gamma
- min_epsilon
- decay_episodes 
- target_update_period 
- target_update_tau 
- batch_size 
- collect_steps_per_episode 
- memory_length
- collect_every 
- n_step_update

```
model = TrainDDQN(episodes, warmup_steps, learning_rate, gamma, min_epsilon, decay_episodes,    target_update_period=target_update_period, target_update_tau=target_update_tau, batch_size=batch_size, collect_steps_per_episode=collect_steps_per_episode, memory_length=memory_length, collect_every=collect_every, n_step_update=n_step_update)
```
### ```TrainDDQN``` includes several methods 

### Example codes: 
```
model.fit(X_train, y_train, layers)
model.q_net.summary() #print out the training layers information
model.train(X_val, y_val, "Precision")

stats = model.evaluate(X_test, y_test, X_train, y_train)
y_pred = model.predict(X_test) 
y_pred_q = model.predict_proba(X_test)
print(stats)
print(y_pred)
print(y_pred_q)
```




## 5. Available Function and Syntax

| Functions         | Purpose       |
| ------------ |------------ |
| ``` fit() ```      | Initializes the neural networks, DDQN-agent, collect policies and replay buffer.       |
| ```train()```      | Starts the training of the model. Includes warmup period, metrics collection and model saving.    |  
| ```evaluate()```        | Evaluate the trained Q-network on test set   | 
| ```predict()```       | Return the prediction output      |  
| ```predict_proba()```       | Return the q values of each label; <span style = "color:red">Similar to ```sklearn.predict_proba()```, however the output is not probability, but corresponding q values</span>      | 

### Fit reinforcement learning models
<div class="alert">fit(X,y, layers, imb_ratio: float = None,  loss_fn=common.element_wise_squared_loss )</div>

|          |        | 
| ------------ |------------ |
|  Parameters    |  X: np.array; training sample ; <br />y: np.array; training labels<br />layers: Previously defined training layers|
|  Return    |self: returns an instance of self. |


### Train reinforcement learning models
<div class="alert">train(X,y, save_best ) </div>

|          |        | 
| ------------ |------------ |
|  Parameters    |  X: np.array; training or validation sample ; <br />y: np.array; training labels or validation labels<br />save_best: Choose one of: { F1, Precision, Recall, TP, TN, FP, FN}<br/>  <span style="color:red">**The algorithm will choose the best model based on the selected metric</span>|
|  Return    |self: returns an instance of self. |


### Predict and Evaluate reinforcement learning models
<div class="alert">evaluate(X_test,y_test, X_train, y_train)</div>

|          |        | 
| ------------ |------------ |
|  Parameters    | X_test: np.array; test sample; <br/>y_test:np.array; test labels <br/> X_train: np.array; training  sample ; <br />y_train: np.array; training labels|
|  Return    |Roc curve, PR curve, {F1, Precision, Recall, TP, TN, FP, FN} |

### Predict the class labels for the provided data.
<div class="alert">predict(X_test)</div>

|          |        | 
| ------------ |------------ |
|  Parameters    | X_test: np.array; test sample|
|  Return |   Class labels for each data sample, 0 or 1 |

### Return probability estimates for the test data X.
<div class="alert">predict(X_test)</div>

|          |        | 
| ------------ |------------ |
|  Parameters    | X_test: np.array; test sample|
|  Return    |The class q-values of the input samples |

<marquee direction="right">&lt;&gt;&lt;&nbsp;&hellip;</marquee>



# Appendix: How does Deep Q Network work? 
This section will give a detailed explanation of deep q network and its algorithm 

 
## [Environment](https://dataiku.gbx.novartis.net/projects/DS_AI_INNOV_CODES/libedition) 
In RL, an environment the task or problem to be solved, where the agent interacts with. The environment usually consists of several elements: 
1. State : The state of the environment is determined by the trainin gsample. We view the first sample $`x_1`$ as the initial state. We use ```ArraySpec``` to take in the shape of the data frame for observation; The current state is the the current sample (i.e row) 
2. Action space: If two prediction outcomes, action = {0,1}; We use ```BoundedArraySpec``` to specify 0 and 1 
3. Reward and Punishment: If predict correctly, reward; Otherwise, punish; In imbalanced dataset, we assign different reward and punishment values based on whether minority or majority. We define reward and punishment in the function ```step()``` and update the episode when the the current prediction ends. 
4.  Episode: Transition trajectory from the initial state to the terminal state. An episode ends when all samples in training data set are classified or when the agent misclassifies the sample from minority class. 
5.  Policy: A policy is a mapping from perceived states of the environment to actions to be taken when in those states. $`\pi_{\theta}`$

## [Agent](https://dataiku.gbx.novartis.net/projects/DS_AI_INNOV_CODES/notebooks/jupyter/RL_TRAINING/)
The algorithm used to solve an RL problem is represented by an ```Agent``` . We use Deep Q-network.  ```QNetwork```, a neural network model that can learn to predict  ```QValues```(expected returns) for all actions, given an observation from the environment. 

We use  ```tf_agents.networks``` to create a  ```QNetwork```, which consists of a sequence of  ```tf.keras.layers.Dense```layers, where the final layer will have 1 output for each possible action. 

Then, we use  ```tf_agents.agents.dqn.dqn_agent``` to instantiate a  ```DqnAgent```. We use  ```AdamOptimizer``` for an optimizer, and ```common.element_wise_squared_loss``` as loss function. 

## Policy: Evaluation and deployment 
A policy defines the way an agent acts in an environment. The goal is to train the underlying model until the policy produces the desired outcome. 
In our case, the desired outcome is keeping predicting right label. Use  ```tf_agents.policies.random_tf_policy``` to create a policy which will randomly select an action for each time step.  

## Replay Buffer 
The replay buffer keeps track of data colelcted from the environment. We use ```tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer``` which requires the specs for data collected. 

## Training
1. Create an environment 
2. Create an agent 
3. Collect data 

Sequential Model: Train on dense layer with 256 units/neurons and pass on Relu activation function. Output layer only has 2 units which is the number of classification output  without activation. It just has the Q value. 
```
layers = [Dense(256, activation="relu"), Dropout(0.2),
          Dense(256, activation="relu"), Dropout(0.2),
          Dense(2, activation=None)]```

To define the parameters to train the model, we use function```compile_model()``` which initialize the neural networks, DQN-agent, policies and replay buffer defined above. 

**Important notes :zap: ** 
When training is finished, it will save the Q-network as pickle file as **%Y%m%d_%H%M%S.pkl** under folder **models**.  So make sure you have the models folder set up before training the model. 
## Prediction
It computes y_pred using the given network saved 

Computes metrics using y_true and y_pred containing F1, precision, recall, TP, TN,FP,FN using ```classification_metrics```. 
```plot_pr_curve``` plots the PR curve of X_test and y_test.  Train AP is calculated by ```sklearn.metrics.average_precision_score (y_train, y_val_score)``` which calculates the average precision (AP) for prediction scores. AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold with the increase in recall from the previous threshold used as the weight: 
```math
{\displaystyle AP = \sum_n(R_n - R_{n-1})P_n .}
```
where  $`P_n, R_n`$ are the precision and recall at the nth threshold. Test AP is calculated by ```sklearn.metrics.average_precision_score (y_test, y_test_score)```

```plot_roc_curve``` plots the ROC curve of X_test and y_test. Train AP is calculated by ```sklearn.metrics.roc_curve(y_train, y_train_score)```. Test AP is calculated by ```sklearn.metrics.roc_curve(y_test, y_test_score)```. 


## Reference: 
 _ Deep Reinforcement Learning for Imbalanced Classification_ : https://arxiv.org/abs/1901.01379
 
 _imbDRL_ : https://github.com/Denbergvanthijs/imbDRL


## new line added. 07/09/2023
