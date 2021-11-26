import os
import cv2
from tqdm import tqdm

directory = './video_result_demo/'
if not os.path.exists(directory):
    os.makedirs(directory)

vis_dir = './demo_videos/'
for v_dir in tqdm(os.listdir(vis_dir)):
    video_list = os.listdir(vis_dir + v_dir + '/')
    video_list.sort()
    video_name = directory + v_dir + '.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #video = cv2.VideoWriter(video_name, 0x7634706d, 5.0, (1920, 1080))
    video = cv2.VideoWriter(video_name, fourcc, 5.0, (1920, 1080))

    for f in video_list:
        img_fig = cv2.imread(vis_dir + v_dir + '/'+f)
        video.write(img_fig)

    video.release()



