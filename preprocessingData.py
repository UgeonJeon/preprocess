import re, os, time, datetime, random, cv2, shutil, math, sys, scipy.spatial

import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
import os.path as osp


# annotation
def verify_annotation(annotation):
    if type(annotation) != np.ndarray:
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
    

def verify_annotation_file(file_path, save_path = False):
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


def verify_annotation_folder(folder_path, save_path = False):
    empty, error, revise = [], [], []
    annotation_files = glob(folder_path + '/**/*.txt', recursive=True)
    
    pbar = tqdm(annotation_files)
    for file_path in pbar:
        pbar.set_description(f"empty {len(empty)}, revised {len(revise)}, deleted {len(error)}")
        rtn = verify_annotation_file(file_path, save_path)
        if rtn == 'empty':
            empty.append(file_path)
        elif rtn == 'error':
            error.append(file_path)
        elif type(rtn) == np.ndarray:
            revise.append(file_path)
    print('empty {}, revised {}, deleted {}'.format(len(empty), len(revise), len(error)))
    return empty, error, revise



# 내시경 Crop
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


def crop_coordinate(image, threshold = 25):
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
#     max_area = image.shape[0] * image.shape[1] # * 0.99

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
        # adjust width and height
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
        
    else :
        return 'None'
    
    if (y_max - y_min)/(x_max - x_min) > 1 or (y_max - y_min)/(x_max - x_min) < 0.729:
        return 'None' # 비율 이상한 이미지들 받아 재처리하기 위함
    else:
        
        return x_min, x_max, y_min, y_max
    

# 이미지에서 내시경 부분만 Crop
def crop_Endoscopic_image(file_path, save_path = False, revise_label = False, return_image = False, threshold = 25):
    """
    read image and return cropped image & annotation file
    """
    
    image = cv2.imread(file_path)
    old_width, old_height = image.shape[1], image.shape[0]
    base_name = os.path.basename(file_path)
    file_format = base_name.split('.')[-1]
    
    x1 = "N"
    while type(x1) != int:
        x1, x2, y1, y2 = crop_coordinate(image, threshold)
        threshold += 10
        
        if threshold >= 75:
            threshold = 10
            x1, x2, y1, y2 = crop_coordinate(image, threshold)
            break
    if type(x1) != int:
        return 'error'
    else:
        new_width, new_height = x2-x1, y2-y1
        cropped_image = image[y1:y2,x1:x2]

        if revise_label:
            with open(file_path.replace(f'.{file_format}','.txt')) as f:
                annotation = np.asarray(list(map(lambda x: np.asarray(np.float64(x.strip().split( ))), f.readlines())))
                annotation[:,1::2] = annotation[:,1::2] * old_width
                annotation[:,2::2] =  annotation[:,2::2] * old_height 

            with open(save_path + base_name.replace(f'.{file_format}','.txt'), 'w') as g:
                new_annotation = np.zeros(len(annotation)*5).reshape(len(annotation), 5)
                new_annotation[:,0] = annotation[:,0]
                new_annotation[:,1] = (annotation[:,1] - x1 )/ new_width
                new_annotation[:,2] = (annotation[:,2] - y1) / new_height
                new_annotation[:,3] = annotation[:,3] / new_width
                new_annotation[:,4] = annotation[:,4] / new_height
                #print('before', new_annotation)
                status, new_annotation = verify_annotation(new_annotation)
                #print('after', new_annotation)
                
                if status == 'error':
                    raise Exception("Invalid Annotation")
                else:
                    new_annotation = re.sub(' +',' ',str(new_annotation).replace('[','').replace(']','').replace('\n ','\n'))
                    g.writelines(new_annotation)
                
        if save_path:
            cv2.imwrite(save_path + base_name, cropped_image)
    
    if return_image:
        return cropped_image    

# 비디오에서 내시경 부분만 Crop
def crop_Endoscopic_video_check(file_path, threshold = 25):
    """
    file_path : directory to read video
    save_path : directory to save extracted image
    return cropped video
    원본화질 유지 >> -c:v copy -preset fast -crf 21
    """
    vd_name = os.path.basename(file_path)
    vidcap = cv2.VideoCapture(file_path)
    fps = int(np.ceil(vidcap.get(cv2.CAP_PROP_FPS)))
    
    if vidcap.isOpened() == False:
        raise Exception(f'Cannot Read Video File {file_path}')
    
    success,image = vidcap.read()
    count = 1
    x1 = "N"
    while success:
        if success and count >= fps * 5 and type(x1) != int: # 5초 이후에서 crop 좌표 찾기
            x1, x2, y1, y2 = crop_coordinate(image, threshold)
            
            if threshold >= 75:
                threshold = 10
                x1, x2, y1, y2 = crop_coordinate(image, threshold)
                break
            if type(x1) != int:
                threshold += 10
                
        count += 1
        
        if type(x1) == int:
            break
            
        success,image = vidcap.read()
    
    cropped_image = image[y1:y2, x1:x2]

    return x1, x2, y1, y2

def crop_Endoscopic_video(file_path, save_path, save_format = '.mp4', threshold = 25, resize_x = False, resize_y = False):
    """
    file_path : directory to read video
    save_path : directory to save extracted image
    return cropped video
    원본화질 유지 >> -c:v copy -preset fast -crf 21
    """
    vd_name = os.path.basename(file_path)
    if save_format[0] != '.':
        save_format += '.' + save_format
    new_vd_name = vd_name.replace('-','_').strip()[:-4] + save_format
    vidcap = cv2.VideoCapture(file_path)
    fps = int(np.ceil(vidcap.get(cv2.CAP_PROP_FPS)))
    
    if vidcap.isOpened() == False:
        raise Exception(f'Cannot Read Video File {file_path}')
    
    save_fullPath = os.path.join(save_path, new_vd_name)
    firstFrame_svPath = os.path.join(save_path, 'firstFrame')
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
        
    if os.path.isdir(firstFrame_svPath) == False:
        os.mkdir(firstFrame_svPath)
    
    success,image = vidcap.read()
    count = 1
    x1 = "N"
    while success:
        if success and count >= fps * 5 and type(x1) != int: # 5초 이후에서 crop 좌표 찾기
            x1, x2, y1, y2 = crop_coordinate(image, threshold)
            
            if threshold >= 75:
                threshold = 10
                x1, x2, y1, y2 = crop_coordinate(image, threshold)
                break
            if type(x1) != int:
                threshold += 10
                
        count += 1
        
        if type(x1) == int:
            break
            
        success,image = vidcap.read()
    
    cropped_image = image[y1:y2, x1:x2]
    cv2.imwrite(firstFrame_svPath + '/' + new_vd_name[:-4] + '.jpg', cropped_image)
    
    if resize_x * resize_y:
        raise ValueError(f"boolean of resize_x, resize_y don't match")
    elif not resize_x and not resize_y:
        os.system(f'ffmpeg -y -i {file_path} -c:v copy -preset fast -an -filter:v "crop={x2-x1}:{y2-y1}:{x1}:{y1}"  {save_fullPath}') 
    elif resize_x and resize_y:
        assert type(resize) == int, 'resize should be int'
        os.system(f'ffmpeg -y -i {file_path} -s {resize_x}:{resize_y} -c:v copy -preset fast -an -filter:v "crop={x2-x1}:{y2-y1}:{x1}:{y1}"  {save_fullPath}') 
        
    return x1, x2, y1, y2


# 원하는 구간 Clip
def extractClip(file_path, save_path, save_filename, start, end, sec_earier = 0, sec_later = 0, 
                resize_x = False, resize_y = False, save_format = '.mp4'):
    """
    file_path : directory to read video
    save_path : directory to save extracted image
    save_filename = name you want to save the file
    sec_earier : advancing clip start point to n seconds
    sec_later : postponing clip end point to n seconds
    resize_x, resize_y = int if you want to resize the video
    """

    video_format = os.path.basename(file_path).split('.')[-1]
    if save_format[0] != '.':
        save_format += '.' + save_format
        
    save_fullPath = os.path.join(save_path, f"{save_filename}{save_format}")
    if isinstance(start, datetime.time) == False:
        start = datetime.datetime.strptime(start, '%H:%M:%S').time()
    if isinstance(end, datetime.time) == False:
        end = datetime.datetime.strptime(end, '%H:%M:%S').time()

    if start <= datetime.datetime(100, 1, 1, 0, 0, sec_earier).time():
        start = datetime.datetime(100, 1, 1, 0, 0, 0).time()
    else:
        start = (datetime.datetime(100, 1, 1, start.hour, start.minute, start.second) - datetime.timedelta(seconds = sec_earier)).time()
    end = (datetime.datetime(100, 1, 1, end.hour, end.minute, end.second) + datetime.timedelta(seconds = sec_later)).time()
       
    if resize_x * resize_y:
        raise ValueError(f"boolean of resize_x, resize_y don't match")
    elif not resize_x and not resize_y:
        os.system(f"ffmpeg -y -i {file_path} -c:v copy  -preset fast -ss {start} -to {end} {save_fullPath}")
    elif resize_x and resize_y:
        assert type(resize) == int, 'resize should be int'
        os.system(f'ffmpeg -y -i {file_path} -s {resize_x}:{resize_y} -c:v copy -preset fast -ss {start} -to {end} {save_fullPath}')
        
    
# extraction
def frameExtraction(file_path, save_path, freeze_detect = 0, frame_interval = 1, freeze_time_thresh = 0.3, save_format = 'jpg'):
    """
    file_path : directory to read video
    save_path : directory to save extracted image
    frame_interval : extract all Frame if 1, else extract image every n frame
    freeze_detect
      0 : only frame extraction
      1 : only freeze detection
      2 : frame extraction + freeze detection
    save_format : format to save images; 'jpg' or 'png'
    """
    read_path = os.path.dirname(file_path)
    video_name = os.path.basename(file_path).split('.')[0]
    video_format = os.path.basename(file_path).split('.')[-1]
    save_format = save_format.replace('.', '')
    
    save_path = os.path.join(save_path, video_name)
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)

    vidcap = cv2.VideoCapture(f'{read_path}/{video_name}.{video_format}')
    fps = int(np.ceil(vidcap.get(cv2.CAP_PROP_FPS)))
    
    if vidcap.isOpened() == False:
        raise Exception(f'Cannot Read Video File {file_path}')
    
    # only extract png
    if freeze_detect == 0:
        extract_onlyFrame(video_name, vidcap, save_path, frame_interval, save_format)
    # only freeze detect
    elif freeze_detect == 1:
        extract_onlyFreeze(video_name, vidcap, save_path, freeze_time_thresh, save_format)
    # extract both
    elif freeze_detect == 2:
        extract_both(video_name, vidcap, save_path, frame_interval, freeze_time_thresh, save_format)
        
        
def dhash_image(image, hash_size = 32):
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


def extract_onlyFrame(video_name, vidcap, save_path, frame_interval, save_format):
    # print('extracting only png images...')
    # initialize
    frame_index = 1 # 프레임 수
    save_index = 1 # 저장 인덱스
    success,image = vidcap.read()
    
    ext_save_path = os.path.join(save_path,'image_extraction')
    if os.path.isdir(ext_save_path) == False:
        os.mkdir(ext_save_path)

    if frame_interval == 1:
        while success:
            success,image = vidcap.read()
            if success:
                saving_filename = f"{video_name}_{str(int(save_index)):0>5}.{save_format}"
                cv2.imwrite(os.path.join(ext_save_path,saving_filename), image)
                save_index += 1
            frame_index += 1
    else:
        assert type(frame_interval) == int and frame_interval > 1, 'Invalid frame_interval input'
        while success:
            success,image = vidcap.read()
            if success and frame_index % frame_interval == 0:
                saving_filename = f"{video_name}_{str(int(save_index)):0>5}.{save_format}"
                cv2.imwrite(os.path.join(ext_save_path,saving_filename), image)
                save_index += 1
            frame_index += 1
            
            
def extract_onlyFreeze(video_name, vidcap, save_path, freeze_time_thresh, save_format):
    fps = int(np.ceil(vidcap.get(cv2.CAP_PROP_FPS)))
    
    frz_save_path = os.path.join(save_path,'freeze_image')
    if os.path.isdir(frz_save_path) == False:
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
        success,image = vidcap.read()
        if success:
            new_hash = dhash_image(image, 32)
            distance = scipy.spatial.distance.hamming(prev_hash, new_hash)

            if distance < 0.05:
                frz_frame_list.append(image)

            else: # freeze 빠져나왔을 때
                if len(frz_frame_list)+1 >= frz_frame_thresh: # 0.1초 이상이면 freeze로 인식
                    # for save_index, frz_frame in enumerate(frz_frame_list):
                    #     saving_filename = f"{video_name}_frz{freeze_index}_{str(save_index+1):0>2}.{save_format}"
                    #     cv2.imwrite(os.path.join(frz_save_path, saving_filename), frz_frame)
                    frz_save_index = frame_index-1
                    saving_filename = f"{video_name}_frz{freeze_index}_{str(frz_save_index):0>5}.png"
                    cv2.imwrite(os.path.join(frz_save_path, saving_filename), frz_frame_list[0])
                    
                    temp_frz = {freeze_index : {'frame_idx' : frame_index-1, 
                                                'length': len(frz_frame_list), 
                                                'file_name':saving_filename}}
                    temp_frz_dict.update(temp_frz)
                    freeze_index += 1
                frz_frame_list.clear()

            prev_image = image
            prev_hash = new_hash
            frame_index += 1
    freeze_dict[video_name] = temp_frz_dict

    # save
    rst_df = pd.DataFrame(columns = ['video_name', 'frz_idx','frame_idx','length', 'file_name'])
    for k, v in freeze_dict.items():
        temp_df = pd.DataFrame.from_dict(v, orient='index').reset_index().rename(columns ={'index': 'frz_idx', 
                                                                                           'frame_': 'frame_idx'})
        temp_df['video_name'] = k
        rst_df = pd.concat([rst_df,temp_df])

    rst_df.to_csv(os.path.join(frz_save_path, 'freeze.csv'), index=False)
    
    
def extract_both(video_name, vidcap, save_path, frame_interval, freeze_time_thresh, save_format):
    fps = int(np.ceil(vidcap.get(cv2.CAP_PROP_FPS)))
    ext_save_path = os.path.join(save_path,'image_extraction')
    frz_save_path = os.path.join(save_path,'freeze_image')
    if os.path.isdir(ext_save_path) == False:
        os.mkdir(ext_save_path)
    if os.path.isdir(frz_save_path) == False:
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
        success,image = vidcap.read()
        if success:
            new_hash = dhash_image(image, 32)
            distance = scipy.spatial.distance.hamming(prev_hash, new_hash)

            if distance < 0.05:
                frz_frame_list.append(image)
            else: # freeze 빠져나왔을 때
                if len(frz_frame_list)+1 >= frz_frame_thresh: # freeze로 인식
                    frz_save_index = frame_index-1
                    saving_filename_frz = f"{video_name}_frz_{str(frz_save_index):0>5}.{save_format}"
                    cv2.imwrite(os.path.join(frz_save_path,saving_filename_frz), frz_frame_list[0])

                    temp_frz = {freeze_index : {'frame_idx' : frz_save_index, 'length': len(frz_frame_list), 
                                                'file_name':saving_filename_frz}}
                    temp_frz_dict.update(temp_frz)
                    freeze_index += 1
                frz_frame_list.clear()

            prev_image = image
            prev_hash = new_hash

            if frame_index % frame_interval == 0:
                saving_filename = f"{video_name}_{str(int(save_index)):0>5}.{save_format}"
                cv2.imwrite(os.path.join(ext_save_path,saving_filename), image)
                save_index += 1

            frame_index += 1
    freeze_dict[video_name] = temp_frz_dict

    # save
    rst_df = pd.DataFrame(columns = ['video_name', 'frz_idx','frame_idx','length', 'file_name'])
    for k, v in freeze_dict.items():
        temp_df = pd.DataFrame.from_dict(v, orient='index').reset_index().rename(columns ={'index': 'frz_idx', 
                                                                                           'frame_': 'frame_idx'})
        temp_df['video_name'] = k
        rst_df = pd.concat([rst_df,temp_df])

    rst_df.to_csv(os.path.join(frz_save_path, 'freeze.csv'), index=False)
    
# def make_annotation(folder_path, data_path, cfg_path, weight_path, thresh):
#     import cv2
#     from glob import glob 
#     import os.path as osp
#     os.chdir('/home/ugeon/endoai/darknet/')

#     images = glob(f'{folder_path}**/*.jpg', recursive=True)
#     print('images to infer :', len(images))

#     with open('temp_path.txt', 'w') as f:
#         for image in images:
#             f.write(f'{image}\n')
            
#     Data = data_path
#     Cfg  = cfg_path 
#     Weights = weight_path

#     Thresh = f'-thresh {thresh}'

#     command = './darknet detector test ' + Data +' '+ Cfg + ' ' + Weights +' '+ Thresh + ' -dont_show -ext_output < temp_path.txt > temp_result.txt -save_labels '
#     commands = [command]
#     os.system(' '.join(commands))