import cv2
import glob
from tqdm import tqdm
import os

VIDEO_PATH = './data/YouTubeClips/*.avi'


'''Convert Video to Image'''
# Input: video name
# Output: Create a image folder and put the image generated from video to the folder
# i.e. "/images/nameimage1.jpg"
# name - video name, 1 - video image counter, image - a flag to symbolize it is a image
def video_to_image(video_name):
    if not os.path.exists("./Images"):
        os.mkdir("./Images")
    vidcap = cv2.VideoCapture(video_name)

    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = vidcap.read()
        if hasFrames:
            cv2.imwrite("./Images/" + video_name.split('\\')[-1][:-4] + "image" + str(count) + ".jpg", image)
        return hasFrames

    sec = 0
    frameRate = 1 
    count=1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)

if __name__ == "__main__":
    # Usage: download dataset from https://www.cs.utexas.edu/users/ml/clamp/videoDescription/, create a data folder
    # and put both caption & video below the data folder.
    files = glob.glob(VIDEO_PATH)

    for file in tqdm(files):
        video_to_image(file)
