import cv2
from scripts.yunet import YuNet
import numpy as np
from cvzone.Utils import overlayPNG
import random
import time
import os


# Cascade Model
# face_detection_yunet_2023mar_int8.onnx is for far face detection
model = YuNet(modelPath="classifiers/face_detection_yunet_2023mar.onnx",
              inputSize=[320, 320],
              confThreshold=0.9,
              nmsThreshold=0.3,
              topK=5000,
              backendId=3,
              targetId=0)


# Get BBOX Size
def getSize(results):
    positions = []
    for result in results:
        bbox = result[0:4].astype(np.int32)
        positions.append(bbox)

    return positions


# Check High Score
def checkHighScore():
    if score > highScore:
        globals()["highScore"] = score

    open(".highscore", "w").write(str(highScore))


# Add Text to Center
def addTextToCenter(image, text, font=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA, custom_x=None, custom_y=None):
    image = image.copy()
    # Get text dimensions
    text_width, text_height = cv2.getTextSize(
        text, font, fontScale, thickness)[0]

    # Calculate center coordinates
    img_center_x = int(FrameWidth / 2)
    img_center_y = int(FrameHeight / 2)

    text_x = img_center_x - int(text_width / 2)
    text_y = img_center_y + int(text_height / 2)

    if not (custom_x == None):
        text_x = custom_x

    if not (custom_y == None):
        text_y = custom_y

    image = cv2.putText(image, text, (text_x, text_y),
                        font, fontScale, color, thickness, lineType)

    return image


# Update the Position of `Head Bat`
def updateBat(image, face):
    image = image.copy()

    x, y, w, h = face
    foreheadHeight = round(h * 0.25)  # Estimated 25% of the Face
    batHeight = 10
    batX1, batY1 = x, y + foreheadHeight-batHeight
    batX2, batY2 = x+w, y + foreheadHeight

    image = cv2.rectangle(image, (batX1, batY1),
                          (batX2, batY2), (0, 255, 255), -1)

    globals()["batPosX"] = [batX1, batX2]
    globals()["batPosY"] = [batY1, batY2]

    return image


# Update the Position of `Head Bat`
def updateBall(image):
    image = image.copy()

    image = overlayPNG(image, ballImg, (ballPosX, ballPosY))

    return image


# Add Border to Frame - Add instructions - Add Score
def updateRest(image):
    img = image.copy()
    # Draw Bottom Rectangle
    mask = cv2.rectangle(img, (0, 0),
                         (gameBoardWR, FrameHeight), (0, 0, 0), -1)

    mask = cv2.rectangle(mask, (FrameWidth - gameBoardWR, 0),
                         (FrameWidth, FrameHeight), (0, 0, 0), -1)
    image = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

    if not gameStarted:
        image = addTextToCenter(image, "Show Face below this Text to Start the Game", color=(
            255, 255, 255), thickness=3, custom_y=ballPosY*2)

    # Set Score and High Score
    image = cv2.putText(image, "Score: " + str(score), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    image = cv2.putText(image, "High: " + str(highScore), (FrameWidth-200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    return image


# Const Variables
FrameWidth, FrameHeight = 960, 540
gameBoardWR = 250  # Width Reduce -> To het a Phone Camera like Window

# Global Variables
batPosX = [0, 0]
batPosY = [0, 0]
gameOver = False
gameStarted = False
score = 0
highScore = int(open(".highscore", "r").read()
                ) if os.path.isfile(".highscore") else 0
speedUpDelay = 8

# Ball
ballImg = cv2.imread("./asset/ball.png", cv2.IMREAD_UNCHANGED)

ballWH = 50
ballPosX, ballPosY = round((FrameWidth / 2) - (ballWH / 2)), 100
ballDropSpeed = 25
ballSpeedX, ballSpeedY = random.choice([5, -5]), ballDropSpeed

# Capture Video

cap_vid = cv2.VideoCapture(1)
cap_vid.set(cv2.CAP_PROP_FRAME_WIDTH, FrameWidth)
cap_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, FrameHeight)
model.setInputSize([FrameWidth, FrameHeight])

# Timers
speedUpdatedAt = time.time()
faceShownAt = None
while cap_vid.isOpened():
    ret, frame = cap_vid.read()
    if not ret:
        continue

    img = cv2.flip(frame, cv2.CAP_PROP_XI_DECIMATION_HORIZONTAL)

    img = updateBall(img)
    img = updateRest(img)

    results = model.infer(img)
    faces = getSize(results)
    if (len(faces) > 0):
        # The face should be below the ball in order to start playing
        if (not gameStarted) and (faces[0][1] > (ballPosY * 2)) and (faceShownAt is None):
            faceShownAt = time.time()
        img = updateBat(img, faces[0])

    # Show the Text that game is Starting in 2 seconds and Start the game after 2 seconds
    if not (gameStarted):
        if (faceShownAt is not None):
            if ((time.time() - faceShownAt) > 2):
                gameStarted = True
                faceShownAt = None
            else:
                img = addTextToCenter(img, "Starting Game in 2 Seconds",
                                      color=(255, 255, 255), thickness=3, custom_y=80)
                cv2.imshow("Bounce Ball", img)

    # Change Direction when Wall Hits
    if (ballPosX <= gameBoardWR) or (ballPosX+ballWH >= FrameWidth-gameBoardWR):  # Hit the Walls
        ballSpeedX = -ballSpeedX
    if (ballPosY <= 0):  # Hit the Top
        ballPosY = 10
        ballSpeedY = -ballSpeedY
        ballSpeedX = random.choice([ballSpeedX, -ballSpeedX])
    elif (ballPosY >= FrameHeight):  # Down in the Hell!
        gameOver = True

    # Check if ball hits the bat
    # If ballStartX and ballEndX is inside the Bat's x axis pos and ballStartY-halfOfBallDrop and ballEndY+halfOfBallDrop is inside the Bat's y axis pos: to avoid ballSpeed miss match... Ball Hit the Bat!
    if ((batPosX[0] < (ballPosX) < batPosX[1]) or (batPosX[0] < (ballPosX + ballWH) < batPosX[1])) and (batPosY[0]-(ballDropSpeed//2) < ballPosY+ballWH < batPosY[1]+(ballDropSpeed//2)) and (gameStarted):
        ballSpeedY = -ballSpeedY
        ballSpeedX = random.choice([ballSpeedX, -ballSpeedX])
        ballPosY -= 30
        score += 1
        checkHighScore()

    # GameOver Reset
    if (gameOver):
        gameOver = False
        gameStarted = False
        ballPosX, ballPosY = round((FrameWidth / 2) - (ballWH / 2)), 100
        ballSpeedX, ballSpeedY = random.choice([5, -5]), ballDropSpeed
        img = addTextToCenter(img, "Game Over", fontScale=2,
                              color=(255, 255, 255), thickness=6)
        cv2.imshow("Bounce Ball", img)
        cv2.waitKey(2000)  # 2 Seconds
        score = 0

    # Reset Bat position every time to reduce error
    batPosX = [0, 0]
    batPosY = [0, 0]

    if gameStarted:
        ballPosX += ballSpeedX
        ballPosY += ballSpeedY
        if ((time.time() - speedUpdatedAt) > speedUpDelay):  # Speed up a little bit every N seconds
            ballDropSpeed += 1
            ballSpeedY = ballSpeedY + 1 if ballSpeedY > 0 else ballSpeedY - 1
            speedUpdatedAt = time.time()

    cv2.imshow("Bounce Ball", img)

    if (cv2.waitKey(1) & 0xFF == 27):
        break
    if (cv2.getWindowProperty("Bounce Ball", cv2.WND_PROP_VISIBLE) < 1):
        break

# Release Camera and Windows
cap_vid.release()
cv2.destroyAllWindows()
