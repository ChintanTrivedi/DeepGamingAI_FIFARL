import cv2
import numpy as np
from PIL import Image
import pytesseract as pt
from getkeys import key_check
import time
from grabscreen import grab_screen
from matplotlib import pyplot as plt

# necessary evil
pt.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

paused = False
while True:
    if not paused:
        # get screen capture image and set it to the game window coordinates
        screen = grab_screen(region=None)
        screen = screen[25:-40, 1921:]
        screen_resized = cv2.resize(screen, (780, 480))

        # # the reward meter at top right corner of game screen
        # reward_screen = screen[85:130, 1650:1730]
        # i = Image.fromarray(reward_screen.astype('uint8'), 'RGB')
        # total_reward = pt.image_to_string(i)
        # print(total_reward)

        # the reward meter at top right corner of game screen
        reward_screen = screen[745:775, 600:800]
        # reward_screen = cv2.resize(reward_screen, (800, 450))
        i = Image.fromarray(reward_screen.astype('uint8'), 'RGB')
        total_reward = pt.image_to_string(i)
        print(total_reward)
        # visualize everything
        cv2.imshow('What I see', screen_resized)
        cv2.imshow('r', reward_screen[:, :, 0])
        # print(np.mean(reward_screen[:, :, 0]))
        if cv2.waitKey(25) & 0xff == ord('o'):
            cv2.destroyAllWindows()
            break

    # menu control
    keys = key_check()
    if 'P' in keys:
        if paused:
            paused = False
            print('unpaused!')
            time.sleep(1)
        else:
            print('Pausing!')
            paused = True
            cv2.destroyAllWindows()
            time.sleep(1)
    elif 'O' in keys:
        print('Quitting!')
        cv2.destroyAllWindows()
        break
