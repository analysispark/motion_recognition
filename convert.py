import os
import subprocess

# 동영상이 저장된 폴더에 있는 클래스의 종류와 경로를 얻는다.
dir_path = './data/kendo_videos'
class_list = os.listdir(path=dir_path)
print(class_list)
