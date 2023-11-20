#Import the game
import gym_super_mario_bros
#Import the joypad
from nes_py.wrappers import JoypadSpace
#Import controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

#Setup Game

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

