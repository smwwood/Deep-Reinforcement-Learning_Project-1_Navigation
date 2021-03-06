# Deep-Reinforcement-Learning_Project-1_Navigation

The Navigation project requires training an agent to navigate and collect yellow bananas (while avoiding blue bananas) in a large square world.

## Project Details
### State Space & Action Space
* The state space for the navigation environment is 37 dimensions. The dimensions include the agent's velocity, a ray-based perception of the objects around the forward direction, and the agent's forward direction.
* The action space for the navigation task is 4 discrete actions: 0 = forward, 1 = backward, 2 = left turn, 3 = right turn
* The environment is considered solved when the agent acquires an average score of +13 over 100 consecutive episodes.

## Getting Started
You'll want to start by creating a virtual environment for the project's dependencies and file. In Mac, you can create a virtual environment with the following commands in terminal:

```
conda create --name drlnd python=3.6
source activate drlnd
```

To install dependencies, you'll need to follow the instructions here (https://github.com/openai/gym) to perform a minimal install of OpenAI gym. You'll also need to install the classic control and box2d environments by following the directions provided here (https://github.com/openai/gym#classic-control and https://github.com/openai/gym#box2d respectively).
Next, clone the repository.

```
git clone https://github.com/smwwood/Deep-Reinforcement-Learning_Project-1_Navigation.git
```

You'll also need to clone the repository containing the dependencies. 

```python
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```


## Instructions
To run the code, create an IPython kernel for the virtual environment to run the provided notebook. Make sure before running the code in the notebook to change the kernel to match the drlnd environment using the drop-down kernel menu.

```python
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

Run each code block in the Navigation.ipynb to train a new agent to navigate the environment! The notebook will provide a plot of rewards and save the model weights of your successful agent in checkpoint.pth. Please note that this will save over the saved model weights that are already stored in checkpoint.pth.
