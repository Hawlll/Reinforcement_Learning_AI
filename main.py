#Import the game
import gym_super_mario_bros
#Import the joypad
from nes_py.wrappers import JoypadSpace
#Import controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
#Import framstack and grayscale wrappers
from gym.wrappers.gray_scale_observation import GrayScaleObservation
#import vectorization wrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt
import os
from stable_baselines3.ppo import PPO #algorithm used to train model
from stable_baselines3.common.callbacks import BaseCallback #saving models


#Fixes bug that requires seed as parameter for reset()
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

#Setup Game

env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='human')

#Simplifies movement controls to 7. Allows easier training of model
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#Created flag - restart or not
done = True
#Loop through each frame in game
for step in range(100):
    if done:
        #Start the game
        env.reset()
    #Do random actions
    #State is the current state of the game during each frame in numpy arrays
    #reward is basically validation data (moving right as fast as possible without dying)
    #done is a condition of whether mario is dead or not
    #info is time, position, etc
    state, reward, done, _, info = env.step(env.action_space.sample())
    #Show game on screen
    env.render()
env.close()

#Create the base environment
env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='human')

#Simplify controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#Grayscale (cuts down information of state)
env = GrayScaleObservation(env, keep_dim=True)

#Wrap inside the dummy enviornment (wraps an array)
env = DummyVecEnv([lambda: env])

#Stack the frames
env = VecFrameStack(env, 4, channels_order='last')

#visualize frame stack
# state = env.reset()
# state, reward, done, info = env.step([5])
# plt.figure(figsize=(20,16))
# for index in range(state.shape[3]):
#     plt.subplot(1,4,index+1)
#     plt.imshow(state[0][:,:,index])
# plt.show()

#TrainAndLoggingCallback class saves model

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
    
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'best_model{self.n_calls}')
            self.model.save(model_path)
        return True

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

#Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)

#Train model
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)
model.learn(total_timesteps=1000000, callback=callback)
