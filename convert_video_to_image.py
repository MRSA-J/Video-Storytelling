import cv2
import glob
from tqdm import tqdm
files = glob.glob('./*.avi')
def deal_video(video_name):
    vidcap = cv2.VideoCapture(video_name)
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            cv2.imwrite("../images/"+video_name[:-4]+"image"+str(count)+".jpg", image)
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
for file in tqdm(files):
    deal_video(file)