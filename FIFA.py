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

    def _get_reward(self, action):
        screen = grab_screen(region=None)
        screen = screen[25:-40, 1921:]
        screen_resized = cv2.resize(screen, (780, 480))

        # the reward meter at top right corner of game screen
        reward_screen = screen[85:130, 1650:1730]
        i = Image.fromarray(reward_screen.astype('uint8'), 'RGB')
        try:
            ocr_result = pt.image_to_string(i)
            ingame_reward = int(''.join(c for c in ocr_result if c.isdigit()))

            print('current reward: ' + str(self.reward))
            print('observed reward: ' + str(ingame_reward))
            if ingame_reward - self.reward > 1000:
                # if ball hits the target
                self.reward = ingame_reward
                ingame_reward = 1
            elif self._is_over(action):
                # if ball goes in the net but doesn't hit the target
                self.reward = ingame_reward
                ingame_reward = -1
            else:
                # if ball hasn't been shot yet
                self.reward = ingame_reward
                ingame_reward = 0
            print('q-learning reward: ' + str(ingame_reward))
        except:
            ingame_reward = 0
            print('exception q-learning reward: ' + str(ingame_reward))

        return ingame_reward

    def _is_over(self, action):
        # Check if the ball is still there to be hit. If ball is still present in the screenshot, game isn't over yet.
        # What follows is arguably the most sophisticated way to find that out.
        # screen = grab_screen(region=None)
        # screen = screen[25:-40, 1921:]
        # ball_location = screen[790:830, 940:980]
        # # Check red channel (rgb) for grass or ball using threshold 60
        # is_over = np.mean(ball_location[:, :, 0]) < 60
        # print('is over, ball presence. mean=' + str(np.mean(ball_location[:, :, 0])))
        is_over = True if action in [0, 1] else False
        if is_over:
            print('over')
        return is_over

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
            print('pressing enter, reset reward')
            self.reward = 0
            PressKey(enter)
            time.sleep(0.4)
            ReleaseKey(enter)
            time.sleep(2)
            screen = grab_screen(region=None)
            screen = screen[25:-40, 1921:]

        # process through CNN to get the feature map from the raw image
        state = self.cnn_graph.get_image_feature_map(screen)
        return state

    def act(self, action):
        display_action = ['shoot_low', 'shoot_high', 'left_arrow', 'right_arrow']
        print('action: ' + str(display_action[action]))
        # [ shoot_low, shoot_high, left_arrow, right_arrow ]
        keys_to_press = [[spacebar], [spacebar], [leftarrow], [rightarrow]]
        # need to keep all keys pressed for some time before releasing them otherwise fifa considers them as accidental
        # key presses.
        for key in keys_to_press[action]:
            PressKey(key)
        time.sleep(0.05) if action == 0 else time.sleep(0.2)
        for key in keys_to_press[action]:
            ReleaseKey(key)

        # wait until some time after taking action
        if action in [0, 1]:
            time.sleep(5)
        else:
            time.sleep(1)

        reward = self._get_reward(action)
        game_over = self._is_over(action)
        return self.observe(), reward, game_over

    def reset(self):
        return
