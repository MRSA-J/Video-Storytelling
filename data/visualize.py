
video_list=['0bSz70pYAP0_5_15','-vg3vR86fu0_1_6','60x_yxy7Sfw_1_7','9HDUADeA2xg_3_31']

'''
import os
import cv2
def video_to_image(video_name):
    if not os.path.exists(video_name):
        os.mkdir(video_name)
    vidcap = cv2.VideoCapture(video_name)

    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = vidcap.read()
        if hasFrames:
            cv2.imwrite(video_name+"/" + video_name.split('\\')[-1][:-4] + "image" + str(count) + ".jpg", image)
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

for v in video_list:
    video_to_image(v)
'''
import pickle
f=open('save_test_data_seq.pkl','rb')
data=pickle.load(f)
for d in data:
    if d['video_name'] in video_list:
        fw = open(d['video_name'],'w')
        for ref in d['ref_sentences']:
            print(ref,file=fw)