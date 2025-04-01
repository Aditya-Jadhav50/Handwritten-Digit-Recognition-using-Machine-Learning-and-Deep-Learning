import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

# Window dimensions
WINDOWSIZEX = 640
WINDOWSIZEY = 480

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

BOUNDARYINC = 5
IMAGESAVE = False
PREDICT = True

# Load model
MODEL = load_model("bestmodel.keras")
LABELS = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
    5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
}

# Initialize Pygame
pygame.init()

# Define font
FONT = pygame.font.Font(None, 36)

# Create window
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digit Board")

iswriting = False
image_cnt = 1
number_xcord = []
number_ycord = []

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit(0)

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if number_xcord and number_ycord:
                number_xcord = sorted(number_xcord)
                number_ycord = sorted(number_ycord)

                rect_min_x = max(number_xcord[0] - BOUNDARYINC, 0)
                rect_max_x = min(WINDOWSIZEX, number_xcord[-1] + BOUNDARYINC)
                rect_min_Y = max(number_ycord[0] - BOUNDARYINC, 0)
                rect_max_Y = min(WINDOWSIZEY, number_ycord[-1] + BOUNDARYINC)

                number_xcord = []
                number_ycord = []

                # Get image from Pygame surface
                img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_Y:rect_max_Y].T.astype(np.float32)

                if IMAGESAVE:
                    cv2.imwrite("image.png", img_arr)
                    image_cnt += 1

                if PREDICT:
                    image = cv2.resize(img_arr, (28, 28))
                    image = np.pad(image, (10, 10), 'constant', constant_values=0)
                    image = cv2.resize(image, (28, 28)) / 255.0

                    # Convert image type
                    image = image.astype(np.float32)
                    image = image.reshape(1, 28, 28, 1)

                    label = str(LABELS[np.argmax(MODEL.predict(image))])

                    # Draw the rectangle around the image
                    pygame.draw.rect(DISPLAYSURF, RED, (rect_min_x, rect_min_Y, rect_max_x - rect_min_x, rect_max_Y - rect_min_Y), 3)

                    textSurface = FONT.render(label, True, RED, WHITE)
                    textRecObj = textSurface.get_rect()
                    textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_Y

                    DISPLAYSURF.blit(textSurface, textRecObj)

        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()
