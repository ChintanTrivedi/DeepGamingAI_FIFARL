import numpy as np
import pytesseract as pt
import cv2
from CNN import CNN
from PIL import Image
from grabscreen import grab_screen
from directkeys import *


class FIFA(object):
    """
    This class acts as the intermediate "API" to the actual game. Double quotes API because we are not touching the
    game's actual code. It interacts with the game simply using screen-grab (input) and keypress simulation (output)
    using some clever python libraries.
    """

    cnn_graph = CNN()

    def __init__(self):
        self.reset()

    def _get_reward(self):
        screen = grab_screen(region=None)
        screen = screen[25:-40, 1921:]
        screen_resized = cv2.resize(screen, (780, 480))

        # the reward meter at top right corner of game screen
        reward_screen = screen[85:130, 1650:1730]
        i = Image.fromarray(reward_screen.astype('uint8'), 'RGB')
        try:
            total_reward = int(pt.image_to_string(i))
        except:
            total_reward = 0
        return total_reward

    def _is_over(self, action):
        # true if released shoot, false otherwise
        return action == 1

    def observe(self):
        # get current state s from screen using screen-grab
        screen = grab_screen(region=None)
        screen = screen[25:-40, 1921:]
        # process through CNN to get the feature map from the raw image
        state = self.cnn_graph.get_image_feature_map(screen)
        return state

    def act(self, action):
        if action == 1:
            # press shoot
            PressKey(0x39)
        elif action == 2:
            # press left arrow
            PressKey(0x39)
        elif action == 3:
            # press right arrow
            PressKey(0x39)
        elif action == 4:
            # press left arrow + shoot
            PressKey(0x39)
        elif action == 5:
            # press right arrow + shoot
            PressKey(0x39)

        # wait until some time after taking action
        reward = self._get_reward()
        game_over = self._is_over(action)
        return self.observe(), reward, game_over

    def reset(self):
        self.reward = 0
        screen = grab_screen(region=None)
        screen = screen[25:-40, 1921:]
        screen_resized = cv2.resize(screen, (780, 480))
        # press enter
        PressKey(0x39)
