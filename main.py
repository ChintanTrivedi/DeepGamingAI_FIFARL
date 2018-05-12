import cv2
import numpy as np
from PIL import Image
import pytesseract as pt
from getkeys import key_check
import time
from FIFA import FIFA
from train import train
from grabscreen import grab_screen
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd


def baseline_model(grid_size, num_actions, hidden_size):
    # seting up the model with keras
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(grid_size,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.1), "mse")
    return model


model = baseline_model(grid_size=128, num_actions=6, hidden_size=200)
model.summary()

# necessary evil
pt.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

game = FIFA()

epoch = 5000  # Number of games played in training, I found the model needs about 4,000 games till it plays well
# Train the model
# For simplicity of the noteb
hist = train(game, model, epoch, verbose=1)
print("Training done")


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


def load_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # score = loaded_model.evaluate(X, Y, verbose=0)

# paused = False
# while True:
#     if not paused:
#         # get screen capture image and set it to the game window coordinates
#         screen = grab_screen(region=None)
#         screen = screen[25:-40, 1921:]
#         screen_resized = cv2.resize(screen, (780, 480))
#
#         # the reward meter at top right corner of game screen
#         reward_screen = screen[85:130, 1650:1730]
#         i = Image.fromarray(reward_screen.astype('uint8'), 'RGB')
#         total_reward = pt.image_to_string(i)
#         print(total_reward)
#
#         # visualize everything
#         cv2.imshow('What I see', screen_resized)
#         cv2.imshow('What I get', reward_screen)
#         if cv2.waitKey(25) & 0xff == ord('o'):
#             cv2.destroyAllWindows()
#             break
#
#     # menu control
#     keys = key_check()
#     if 'P' in keys:
#         if paused:
#             paused = False
#             print('unpaused!')
#             time.sleep(1)
#         else:
#             print('Pausing!')
#             paused = True
#             cv2.destroyAllWindows()
#             time.sleep(1)
#     elif 'O' in keys:
#         print('Quitting!')
#         cv2.destroyAllWindows()
#         break
