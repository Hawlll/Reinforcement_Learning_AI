#Import the game
import gym_super_mario_bros
#Import the joypad
from nes_py.wrappers import JoypadSpace
#Import controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
#Import framstack and grayscale wrappers
from gym.wrappers import frame_stack, gray_scale_observation
#import vectorization wrapper

#Setup Game

env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='human')

#Simplifies movement controls to 7. Allows easier training of model
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#Created flag - restart or not
done = True
#Loop through each frame in game
for step in range(100000):
    if done:
        #Start the game
        env.reset()
    #Do random actions
    #State is the current state of the game during each frame
    #reward is basically validation data (moving right as fast as possible without dying)
    #done is a condition of whether mario is dead or not
    #info is time, position, etc
    state, reward, done, _, info = env.step(env.action_space.sample())
    #Show game on screen
    env.render()
env.close()

