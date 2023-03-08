import re, os, time, datetime, random, cv2, shutil, math, sys, scipy.spatial

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from glob import glob
import os.path as osp


# annotation
def verify_annotation(annotation):
    if ~isinstance(annotation, (np.ndarray)):
        annotation = np.asarray(annotation)

    x1, x2 = annotation[:,1] - annotation[:,3]/2, annotation[:,1] + annotation[:,3]/2
    y1, y2 = annotation[:,2] - annotation[:,4]/2, annotation[:,2] + annotation[:,4]/2

    old = np.stack((annotation[:,0], x1, x2, y1, y2), axis = 1)
    wrong_idx = (old[:,1] > 1) | (old[:,2] < 0) | (old[:,3] > 1) | (old[:,4] < 0)

    if sum(wrong_idx) == 0: # 모든 라벨링 정상
        annotation[:,1:]  = np.trunc(annotation[:,1:]*1e4)/1e4
        return 'flawless', annotation
    elif sum(wrong_idx) == len(annotation): # 모든 라벨링 비정상
        return 'error', None
    else:
        new = old[~wrong_idx]
        new[:,1::2] = np.where(new[:,1::2] < 0, 0.0001, new[:,1::2])
        new[:,2::2] = np.where(new[:,2::2] > 1, 0.9999, new[:,2::2])

        result_annotation = np.zeros(len(new)*5).reshape(len(new), 5)
        result_annotation[:,0] = new[:,0]
        result_annotation[:,1] = (new[:,1] + new[:,2])/2
        result_annotation[:,2] = (new[:,3] + new[:,4])/2
        result_annotation[:,3] = (new[:,2] - new[:,1])
        result_annotation[:,4] = (new[:,4] - new[:,3])
        result_annotation[:,1:] = np.trunc(result_annotation[:,1:]*1e4)/1e4
        return 'revised', result_annotation

def verify_annotation_file( file_path, save_path = False):
    with open(file_path, 'r') as f:
        annotation = list(map(lambda x: np.asarray(np.float64(x.strip().split( ))), f.readlines()))

    if len(annotation) == 0:
        return 'empty'
    else:
        status, new_annotation = verify_annotation(annotation)
        if status == 'error':
            return 'error'
        elif status == 'flawless':
            return 'flawless'
        else:
            if save_path:
                with open(osp.join(save_path, osp.basename(file_path)), 'w') as g:
                    new_ = re.sub(' +',' ',str(new_annotation).replace('[','').replace(']','').replace('\n ','\n'))
                    g.writelines(new_)
                return new_annotation
            else:
                return new_annotation

def verify_annotation_folder( folder_path, save_path = False):
    empty, error, revise = [], [], []
    annotation_files = glob(folder_path + '/**/*.txt', recursive=True)

    for file_path in tqdm(annotation_files):
        rtn = verify_annotation_file(file_path, save_path)
        if rtn == 'empty':
            empty.append(file_path)
        elif rtn == 'error':
            error.append(file_path)
        elif type(rtn) == np.ndarray:
            revise.append(file_path)
    print('empty {}, revised {}, deleted {}'.format(len(empty), len(revise), len(error)))
    return empty, error, revise


def adjustCroppedWidth(width):
    if width in range(495, 535):
        return 514
    elif width in range(970, 1010):
        return 990
    elif width in range(1220, 1260):
        return 1240
    else: return width

def adjustCroppedHeight(height):
    if height in range(390, 430):
        return 410
    elif height in range(840, 880):
        return 860
    elif height in range(1060, 1100):
        return 1080
    else: return height

def get_cropCoordinate(image, threshold):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1. hsv filtering
    lower_black = np.array([0, 5, 3]) 
    upper_black = np.array([180, 250, 255])

    mask_black = cv2.inRange(hsv_image, lower_black, upper_black)
    black_bit = cv2.bitwise_and(hsv_image, hsv_image, mask = mask_black)

    # threshold
    h, s, v = cv2.split(black_bit)
    _, img_binary  = cv2.threshold(v, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(img_binary , cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    min_area = image.shape[0] * image.shape[1] * 0.3

    target_contour = []
    for cnt in contours:
        cnt_size = cv2.contourArea(cnt)
        if cnt_size < min_area:
            continue

        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) >= 4:
            (x, y, w, h) = cv2.boundingRect(cnt)
            target_contour.append((x, y, w, h))
    target_contour = list(set(target_contour))

    if len(target_contour)==1:
        x, y, w, h = target_contour[0]
        w = adjustCroppedWidth(w)
        h = adjustCroppedHeight(h)
        x_min, x_max, y_min, y_max = x, x+w, y, y+h
    elif len(target_contour)>1:
        target_contour = np.asanyarray(target_contour)
        max_size_index = (target_contour[:,2] * target_contour[:,3]).argmax()
        x, y, w, h = target_contour[max_size_index]
        w = adjustCroppedWidth(w)
        h = adjustCroppedHeight(h)
        x_min, x_max, y_min, y_max = x, x + w, y, y + h

    return x_min, x_max, y_min, y_max

def cropImage_coordinate(image, threshold = 25):
    x1 = "N"
    while type(x1) != int:
        x1, x2, y1, y2 = get_cropCoordinate(image, threshold)
        threshold += 10

        if threshold >= 75:
            threshold = 10
            x1, x2, y1, y2 = get_cropCoordinate(image, threshold)
            break
    return x1, x2, y1, y2

def cropVideo_coordinate(filePath, threshold = 25):
    vidcap = cv2.VideoCapture(filePath)
    fps = int(np.ceil(vidcap.get(cv2.CAP_PROP_FPS)))

    if vidcap.isOpened() == False:
        raise Exception(f'Cannot Read Video File {filePath}')

    success,image = vidcap.read()
    frame_index = 1
    while success:
        if success and count == int(fps * 5): # 5초 이후에서 crop 좌표 찾기
            x1, x2, y1, y2 = get_cropCoordinate(image, threshold)
            break

        frame_index += 1
        success,image = vidcap.read()

    return x1, x2, y1, y2


class FileProcess:
    def __init__(self, filePath):
        self.filePath = filePath
    
    # 이미지에서 내시경 부분만 Crop
    def cropImage(self, threshold, revise_annotation = False):
        """
        read image and return cropped image & annotation file
        input : image_path
        """
        image = cv2.imread(self.filePath)
        file_format = osp.splitext(self.filePath)[1]
        
        # crop image
        x1, x2, y1, y2 = cropImage_coordinate(image, threshold)
        cropped_image = image[y1:y2,x1:x2]

        # adjust annotation
        if revise_annotation:
            old_width, old_height = image.shape[1], image.shape[0]
            new_width, new_height = x2-x1, y2-y1
            
            with open(self.filePath.replace(file_format,'.txt')) as f:
                annotation = np.asarray(list(map(lambda x: np.asarray(np.float64(x.strip().split( ))), f.readlines())))

            annotation[:,1::2] = annotation[:,1::2] * old_width
            annotation[:,2::2] =  annotation[:,2::2] * old_height
            
            new_annotation = np.zeros(len(annotation)*5).reshape(len(annotation), 5)
            new_annotation[:,0] = annotation[:,0]
            new_annotation[:,1] = (annotation[:,1] - x1 )/ new_width
            new_annotation[:,2] = (annotation[:,2] - y1) / new_height
            new_annotation[:,3] = annotation[:,3] / new_width
            new_annotation[:,4] = annotation[:,4] / new_height

#             ret, new_annotation = ProcessAnnotation.verify_annotation(new_annotation)
            ret, new_annotation = verify_annotation(new_annotation)
            return cropped_image, new_annotation
        else:
            return cropped_image
        
        
    def cropVideo(self, save_path, threshold = 25, save_1stFrame = True, 
                  save_format = '.mp4', resize_x = False, resize_y = False):
        """
        saving cropped video with ffmpeg
        filePath : directory to read video
        save_path : directory to save extracted image
        원본화질 유지 >> -c:v copy -preset fast -crf 21
        """
        vd_name = osp.basename(self.filePath).replace('-','_')
        file_format = osp.splitext(self.filePath)[1]
        if save_format[0] != '.':
            save_format += '.' + save_format
        save_name = vd_name.replace(file_format, save_format)
        os.makedirs(save_path, exist_ok=True)

        x1, x2, y1, y2 = cropVideo_coordinate(self.filePath)
        vidcap = cv2.VideoCapture(self.filePath)
        
        success,image = vidcap.read()         
        if save_1stFrame:
            firstFrame_svPath = osp.join(save_path, 'firstFrame')
            os.makedirs(firstFrame_svPath, exist_ok=True)
            cropped_image = image[y1:y2, x1:x2]
            cv2.imwrite(firstFrame_svPath + '/' + vd_name.replace(file_format, '.jpg'), cropped_image)

        if resize_x * resize_y:
            raise ValueError(f"boolean of resize_x, resize_y don't match")
        elif not resize_x and not resize_y:
            os.system(f'ffmpeg -y -i {self.filePath} -c:v libx264 -preset fast -crf 20  -an -filter:v "crop={x2-x1}:{y2-y1}:{x1}:{y1}" {save_path}/{save_name}') 
        elif resize_x and resize_y:
            assert type(resize) == int, 'resize should be int'
            os.system(f'ffmpeg -y -i {self.filePath} -s {resize_x}:{resize_y} -c:v copy -preset fast -an -filter:v "crop={x2-x1}:{y2-y1}:{x1}:{y1}" {save_path}/{save_name}') 
            

    # 원하는 구간 Clip
    def extractClip(self, save_path, save_name, start, end, sec_earier = 0, sec_later = 0, 
                    save_format = '.mp4', resize_x = False, resize_y = False):
        """
        save_path : directory to save extracted image
        save_name = name you want to save the file
        sec_earier : advancing clip start point to n seconds
        sec_later : postponing clip end point to n seconds
        resize_x, resize_y = int if you want to resize the video
        """

        video_format = osp.basename(self.filePath).split('.')[-1]
        if save_format[0] != '.':
            save_format += '.' + save_format
        
#         save_name = osp.basename(self.filePath).replace(file_format, save_format)

        if not isinstance(start, datetime.time):
            start = datetime.datetime.strptime(start, '%H:%M:%S').time()
        if not isinstance(end, datetime.time):
            end = datetime.datetime.strptime(end, '%H:%M:%S').time()

        if start <= datetime.datetime(100, 1, 1, 0, 0, sec_earier).time():
            start = datetime.datetime(100, 1, 1, 0, 0, 0).time()
        else:
            start = (datetime.datetime(100, 1, 1, start.hour, start.minute, start.second) - datetime.timedelta(seconds = sec_earier)).time()
        end = (datetime.datetime(100, 1, 1, end.hour, end.minute, end.second) + datetime.timedelta(seconds = sec_later)).time()

        if resize_x * resize_y:
            raise ValueError(f"boolean of resize_x, resize_y don't match")
            
        elif not resize_x and not resize_y:
            os.system(f"ffmpeg -y -i {self.filePath} -c:v copy -preset fast -ss {start} -to {end} {osp.join(save_path, save_name+save_format)}")
        elif resize_x and resize_y:
            assert type(resize) == int, 'resize should be int'
            os.system(f'ffmpeg -y -i {self.filePath} -s {resize_x}:{resize_y} -c:v libx264 -preset fast -crf 20 -ss {start} -to {end} {osp.join(save_path, save_name+save_format)}')
            

            
class ImageExtraction:
    def __init__(self, filePath):
        self.filePath = filePath
        
    # extraction
    def frameExtraction(self, save_path, freeze_detect = 0, frame_interval = 1, freeze_time_thresh = 0.3, save_format = '.jpg'):
        """
        self.filePath : directory to read video
        save_path : directory to save extracted image
        frame_interval : extract all Frame if 1, else extract image every n frame
        freeze_detect
          0 : only frame extraction
          1 : only freeze detection
          2 : frame extraction + freeze detection
        save_format : format to save images; '.jpg' or '.png'
        """
        video_name = osp.splitext(osp.basename(self.filePath))[0]

        save_path = osp.join(save_path, video_name)
        if osp.isdir(save_path) == False:
            os.mkdir(save_path)

        vidcap = cv2.VideoCapture(self.filePath)
        fps = int(np.ceil(vidcap.get(cv2.CAP_PROP_FPS)))

        if vidcap.isOpened() == False:
            raise Exception(f'Cannot Read Video File {self.filePath}')

        # only extract png
        if freeze_detect == 0:
            assert type(frame_interval) == int and frame_interval >= 1, 'Invalid frame_interval input'
            self.extract_onlyFrame(video_name, vidcap, save_path, frame_interval, save_format)
        # only freeze detect
        elif freeze_detect == 1:
            self.extract_onlyFreeze(video_name, vidcap, save_path, freeze_time_thresh, save_format)
        # extract both
        elif freeze_detect == 2:
            assert type(frame_interval) == int and frame_interval >= 1, 'Invalid frame_interval input'
            self.extract_both(video_name, vidcap, save_path, frame_interval, freeze_time_thresh, save_format)


    def dhash_image(self, image, hash_size = 32):
        '''
        주어진 (N, N+1) 행렬을 처음 한 열을 제거한 정사각행렬과 마지막 한 열을 제거한 정사각행렬의 같은 위치의 
        원소값(gray_scale)을 비교하여 True 또는 False의 값이 부여된 (N,N)행렬을 flatten을 통해 1차원 행렬(=리스트)로 변환하여 리턴
        '''
        # 이미지를 행렬로 읽어옴
        resized = cv2.resize(image, (hash_size + 1, hash_size))
        # 원하는 사이즈로 줄여줌
        gray_scaled = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # 컬러 -> 흑백으로 데이터 사이즈 줄이기
        diff = image[:, 1:] > image[:, :-1]
        hash_ = diff.flatten()
        return hash_


    def extract_onlyFrame(self, video_name, vidcap, save_path, frame_interval, save_format):
        # initialize
        frame_index = 1 # 프레임 수
        save_index = 1 # 저장 인덱스
        success,image = vidcap.read()

        ext_save_path = osp.join(save_path,'image_extraction')
        if osp.isdir(ext_save_path) == False:
            os.mkdir(ext_save_path)

        if frame_interval == 1:
            while success:
                saving_filename = f"{video_name}_{str(int(save_index)):0>5}{save_format}"
                cv2.imwrite(osp.join(ext_save_path,saving_filename), image)
                save_index += 1
                frame_index += 1
                success,image = vidcap.read()
        else:
            while success:
                if frame_index % frame_interval == 0:
                    saving_filename = f"{video_name}_{str(int(save_index)):0>5}{save_format}"
                    cv2.imwrite(osp.join(ext_save_path,saving_filename), image)
                    save_index += 1
                frame_index += 1
                success,image = vidcap.read()

    def extract_onlyFreeze(self, video_name, vidcap, save_path, freeze_time_thresh, save_format):
        fps = int(np.ceil(vidcap.get(cv2.CAP_PROP_FPS)))

        frz_save_path = osp.join(save_path,'freeze_image')
        if osp.isdir(frz_save_path) == False:
            os.mkdir(frz_save_path)

        # initialize
        prev_image = None
        prev_hash = None
        freeze_dict = dict()
        temp_frz_dict = dict()
        freeze_dict[video_name] = dict()
        freeze_index = 1
        frame_index = 1
        frz_frame_list = []
        frz_frame_thresh = fps * freeze_time_thresh

        # read first frame
        success,image = vidcap.read()
        prev_image = image
        prev_hash = dhash_image(image, 32)

        while success:
            new_hash = dhash_image(image, 32)

            if frame_index >= 2:
                distance = scipy.spatial.distance.hamming(prev_hash, new_hash)
                if distance < 0.05:
                    frz_frame_list.append(image)

                else: # freeze 빠져나왔을 때
                    if len(frz_frame_list)+1 >= frz_frame_thresh: # 0.1초 이상이면 freeze로 인식
                        # for save_index, frz_frame in enumerate(frz_frame_list):
                        #     saving_filename = f"{video_name}_frz{freeze_index}_{str(save_index+1):0>2}.{save_format}"
                        #     cv2.imwrite(osp.join(frz_save_path, saving_filename), frz_frame)
                        frz_save_index = frame_index-1
                        saving_filename = f"{video_name}_frz{freeze_index}_{str(frz_save_index):0>5}{save_format}"
                        cv2.imwrite(osp.join(frz_save_path, saving_filename), frz_frame_list[0])

                        temp_frz = {freeze_index : {'frame_idx' : frame_index-1, 
                                                    'length': len(frz_frame_list), 
                                                    'file_name':saving_filename}}
                        temp_frz_dict.update(temp_frz)
                        freeze_index += 1
                    frz_frame_list.clear()

            prev_image = image
            prev_hash = new_hash
            frame_index += 1
            success,image = vidcap.read()
                
        freeze_dict[video_name] = temp_frz_dict

        # save
        rst_df = pd.DataFrame(columns = ['video_name', 'frz_idx','frame_idx','length', 'file_name'])
        for k, v in freeze_dict.items():
            temp_df = pd.DataFrame.from_dict(v, orient='index').reset_index().rename(columns ={'index': 'frz_idx', 
                                                                                               'frame_': 'frame_idx'})
            temp_df['video_name'] = k
            rst_df = pd.concat([rst_df,temp_df])

        rst_df.to_csv(osp.join(frz_save_path, 'freeze.csv'), index=False)


    def extract_both(self, video_name, vidcap, save_path, frame_interval, freeze_time_thresh, save_format):
        fps = int(np.ceil(vidcap.get(cv2.CAP_PROP_FPS)))
        ext_save_path = osp.join(save_path,'image_extraction')
        frz_save_path = osp.join(save_path,'freeze_image')
        if osp.isdir(ext_save_path) == False:
            os.mkdir(ext_save_path)
        if osp.isdir(frz_save_path) == False:
            os.mkdir(frz_save_path)

        # initialize
        prev_image = None
        prev_hash = None
        freeze_dict = dict()
        temp_frz_dict = dict()
        freeze_dict[video_name] = dict()
        freeze_index = 1
        frame_index = 1
        save_index = 1 
        frz_frame_list = []
        frz_frame_thresh = fps * freeze_time_thresh

        # read first frame
        success,image = vidcap.read()
        prev_image = image
        prev_hash = dhash_image(image, 32)

        while success:
            new_hash = dhash_image(image, 32)
            if frame_index >= 2:
                distance = scipy.spatial.distance.hamming(prev_hash, new_hash)

                if distance < 0.05:
                    frz_frame_list.append(image)
                else: # freeze 빠져나왔을 때
                    if len(frz_frame_list)+1 >= frz_frame_thresh: # 0.1초 이상이면 freeze로 인식
                        # for save_index, frz_frame in enumerate(frz_frame_list):
                        #     saving_filename = f"{video_name}_frz{freeze_index}_{str(save_index+1):0>2}.{save_format}"
                        #     cv2.imwrite(osp.join(frz_save_path, saving_filename), frz_frame)
                        frz_save_index = frame_index-1
                        saving_filename = f"{video_name}_frz{freeze_index}_{str(frz_save_index):0>5}{save_format}"
                        cv2.imwrite(osp.join(frz_save_path, saving_filename), frz_frame_list[0])

                        temp_frz = {freeze_index : {'frame_idx' : frz_save_index, 'length': len(frz_frame_list), 
                                                    'file_name':saving_filename}}
                        temp_frz_dict.update(temp_frz)
                        freeze_index += 1
                    frz_frame_list.clear()

            prev_image = image
            prev_hash = new_hash

            if frame_index % frame_interval == 0:
                saving_filename = f"{video_name}_{str(int(save_index)):0>5}{save_format}"
                cv2.imwrite(osp.join(ext_save_path,saving_filename), image)
                save_index += 1

            frame_index += 1
            success,image = vidcap.read()
        freeze_dict[video_name] = temp_frz_dict

        # save
        rst_df = pd.DataFrame(columns = ['video_name', 'frz_idx','frame_idx','length', 'file_name'])
        for k, v in freeze_dict.items():
            temp_df = pd.DataFrame.from_dict(v, orient='index').reset_index().rename(columns ={'index': 'frz_idx', 
                                                                                               'frame_': 'frame_idx'})
            temp_df['video_name'] = k
            rst_df = pd.concat([rst_df,temp_df])

        rst_df.to_csv(osp.join(frz_save_path, 'freeze.csv'), index=False)