import cv2
import glob
from tqdm import tqdm

'''Convert Video to Image'''
# Input: video name
# Output: i.e. "/images/nameimage1.jpg"
# name - video name, 1 - video image counter, image - a flag to symbolize it is a image
def video_to_image(video_name):
    vidcap = cv2.VideoCapture(video_name)

    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = vidcap.read()
        if hasFrames:
            cv2.imwrite("../images/" + video_name[:-4] + "image" + str(count) + ".jpg", image)
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

# When running, create a data foler.
# Download Youtube Clips from https://www.cs.utexas.edu/users/ml/clamp/videoDescription/ and put it below "data" folder to run
if __name__ == "__main__":
    files = glob.glob('./data/YouTubeClips/*.avi')

    for file in tqdm(files):
        video_to_image(file)
