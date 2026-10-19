import os
import cv2
import glob

video_path = 'Vp.mp4'
#output_folder = './test/Indoor/data/'
output_folder = './test/Indoor/Vp/'
if os.path.isdir(output_folder):
    print("Delete old result folder: {}".format(output_folder))
    os.system("rm -rf {}".format(output_folder))
os.system("mkdir {}".format(output_folder))
print("create folder: {}".format(output_folder))

vc = cv2.VideoCapture(video_path)
fps = vc.get(5)
print(fps)
print(vc.get(3))
print(vc.get(4))
frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
print(frame_count)
video = []
i = 1
for idx in range(frame_count):
    vc.set(1, idx)
    ret, frame = vc.read()
    #height, width, layers = frame.shape
    #size = (width, height)

    if frame is not None:
        file_name = '{}{:08d}.png'.format(output_folder,i)
        i += 1
        cv2.imwrite(file_name, frame)

    print("\rprocess: {}/{}".format(idx+1 , frame_count), '')
vc.release()

def getName(num):
    if num <10:
        strRes = '0000000' + str(num)
    elif num <100:
        strRes = '000000' + str(num)
    elif num <1000:
        strRes = '00000' + str(num)
    elif num < 10000:
        strRes = '0000' + str(num)
    return strRes
 
file_object = open('Vp.txt','w')
Ostr = ''
num = len(os.listdir('test/Indoor/Vp'))
for i in range(1,num+1):
    name = getName(i)
    Ostr = name + "\n"
    file_object.writelines(Ostr)
file_object.close()
print('file compelete')
