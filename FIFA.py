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
    reward = 0

    def __init__(self):
        self.reset()

    def _get_reward(self):
        print('_get_reward')
        screen = grab_screen(region=None)
        screen = screen[25:-40, 1921:]
        screen_resized = cv2.resize(screen, (780, 480))

        # the reward meter at top right corner of game screen
        reward_screen = screen[85:130, 1650:1730]
        i = Image.fromarray(reward_screen.astype('uint8'), 'RGB')
        try:
            total_reward = int(pt.image_to_string(i))
            if total_reward - self.reward > 0:
                self.reward = total_reward
                total_reward = 1
            else:
                total_reward = 0
        except:
            total_reward = 0
        return total_reward

    def _is_over(self):
        print('_is_over')
        # Check if the ball is still there to be hit. If ball is still present in the screenshot, game isn't over yet.
        # What follows is arguably the most sophisticated way to find that out.
        screen = grab_screen(region=None)
        screen = screen[25:-40, 1921:]
        ball_location = screen[790:830, 940:980]
        # Check red channel (rgb) for grass or ball using threshold 60
        return np.mean(ball_location[:, :, 0]) < 60

    def observe(self):
        print('observe')
        # get current state s from screen using screen-grab
        screen = grab_screen(region=None)
        screen = screen[25:-40, 1921:]

        # if drill over, restart drill and take screenshot again
        restart_button = screen[745:775, 600:720]
        i = Image.fromarray(restart_button.astype('uint8'), 'RGB')
        restart_text = pt.image_to_string(i)
        if "RETRV DRILL" in restart_text:
            # press enter key
            print('pressing enter')
            PressKey(enter)
            time.sleep(1)
            ReleaseKey(enter)
            time.sleep(2)
            screen = grab_screen(region=None)
            screen = screen[25:-40, 1921:]

        # process through CNN to get the feature map from the raw image
        state = self.cnn_graph.get_image_feature_map(screen)
        return state

    def act(self, action):
        # [ nothing, shoot, left_arrow, right_arrow, left_arrow_shoot, right_arrow_shoot ]
        keys_to_press = [[], [spacebar], [leftarrow], [rightarrow], [leftarrow, spacebar], [rightarrow, spacebar]]
        # need to keep all keys pressed for some time before releasing them otherwise fifa considers them as accidental
        # key presses.
        for key in keys_to_press[action]:
            PressKey(key)
        time.sleep(0.2)
        for key in keys_to_press[action]:
            ReleaseKey(key)

        # wait until some time after taking action
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self):
        self.reward = 0
