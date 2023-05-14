from re import A, I
from imutils.video import VideoStream
from imutils.video import FPS

import argparse
import cv2
from glob import glob
import json
import progressbar
from time import sleep
import math
import matplotlib.pyplot as plt
import time
import os
from random import randrange
import random
import pandas as pd
import numpy as np
import h5py


#function to save frames from video
def save_frames(video_path, mocap_file, footp_file, save_dir, all_keyframes, sub_takes, crop_size, M, N, subject, take):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # cropped_video = cv2.VideoWriter(save_dir, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (crop_size, crop_size))
    # total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    with h5py.File(mocap_file, 'r') as f:
        joints = np.array(f['POSE'])

    with h5py.File(footp_file, 'r') as f:
        footp = np.array(f['PRESSURE'])

    subject_number = int(subject.partition("Subject")[-1])

    #splicing video function
    # all_keyframes = np.loadtxt(keyframe_labels, delimiter=',')
    # sub_takes = pd.read_csv(sub_take_path)

    bar = progressbar.ProgressBar(maxval=total_frames, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    save_dir_base = os.path.join(save_dir, f"Take_{str(take)}")
    if not os.path.exists(save_dir_base):
        os.makedirs(save_dir_base)

    col = sub_takes.query(f'subject=={subject_number} & take=={take}').index[0]#---change this to pandas
    keyframes = all_keyframes[:, col]
    valid_rand_frame = False
    while not valid_rand_frame:
        rand_keyframe = randrange(0, len(keyframes)-1)
        if not math.isnan(keyframes[rand_keyframe]) and not math.isnan(keyframes[rand_keyframe+1]):
            valid_rand_frame = True

    rand_range = keyframes[rand_keyframe+1] - keyframes[rand_keyframe]
    rand_frame = int(keyframes[rand_keyframe+1] - (rand_range // 2))#CHANGE THIS TO keyframes[rand_keyframe+1] - (N * fps)?
    keyframes = np.insert(keyframes, 0, rand_frame, axis=0)

    # Precompute start and end frames for each keyframe
    current_keyframe_idx = 0# 1 to 45 are sub-poses. 0 is not a sub-pose
    frame_2_keyframe_idx = {}#every frame in range start_frame to end_frame is assigned to a keyframe index
    for keyframe in keyframes:
        if np.isnan(keyframe):
            current_keyframe_idx += 1
            continue
        start_frame = int(keyframe - M * fps)
        end_frame = int(keyframe + N * fps)
        if end_frame > total_frames:
            end_frame = total_frames
        if start_frame < 0:
            start_frame = 0
        try:
            frame_2_keyframe_idx.update({i: current_keyframe_idx for i in range(start_frame, end_frame)})
        except:
            import pdb; pdb.set_trace()
        current_keyframe_idx += 1


    current_frame_idx = 0

    #crop_video function
    prev_w, prev_h, prev_x, prev_y = 0, 0, 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        joint = joints[:,:,current_frame_idx]
        pelvis_loc = joint[:,12]
        x = int(pelvis_loc[0])
        y = int(pelvis_loc[1])

        if math.isnan(x):
            x = prev_x
        if math.isnan(y):
            y = prev_y

        # Adjust for centering and crop
        x = x - (crop_size // 2) 
        y = y - (crop_size // 2)

        # Check for large jumps in pelvis position
        if (abs(x - prev_x) > 50 or abs(y - prev_y)> 50) and current_frame_idx > 500:
            x = prev_x
            y = prev_y
            w = prev_w
            h = prev_h

        if x < 0 or x + crop_size > frame.shape[1]:
            print("x :",x,"||x + crop_size :",x + crop_size,"||frame.shape[1] : ",frame.shape[1])
            diff = abs(x)
            frame = cv2.copyMakeBorder(frame, 0, 0, diff, diff, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if y < 0 or y + crop_size > frame.shape[0]:
            print("y :",y,"||y + crop_size :",y + crop_size,"||frame.shape[0] : ",frame.shape[0])
            diff = abs(y)
            frame = cv2.copyMakeBorder(frame, diff, diff, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if current_frame_idx in frame_2_keyframe_idx:
            crop = frame[y:y+crop_size, x:x+crop_size]
            cv2.imwrite(os.path.join(save_dir_base, f"frame_{current_frame_idx}_class_{frame_2_keyframe_idx[current_frame_idx]}.jpg"), crop)
            mocap = joints[:,:,current_frame_idx]
            mocap = mocap.astype(np.float32)
            fp = footp[:,:,:,current_frame_idx]
            with h5py.File(os.path.join(save_dir_base, f"frame_{current_frame_idx}_class_{frame_2_keyframe_idx[current_frame_idx]}.h5"), 'w') as f:
                f.create_dataset('MOCAP', data=mocap)
                f.create_dataset('PRESSURE', data=fp)

        prev_x = x
        prev_y = y
        bar.update(current_frame_idx)
        current_frame_idx += 1
    bar.finish()
    cap.release()


M = 2
N = 0.4
crop_size  = 500

keyframe_labels = "./frame_labels/taiji_keyframes.csv"
sub_take_path = "./frame_labels/sub_takes.csv"

all_keyframes = np.loadtxt(keyframe_labels, delimiter=',')
sub_takes = pd.read_csv(sub_take_path)

for sub_num in range(1,11):
    subject = "Subject"+str(sub_num)
    for take in range(1,11):
        print("-----------------------------------")
        video_path = "./dataset/Video/"+ subject +"/Video_V1_"+str(take)+".mp4"
        mocap_file = "./dataset/Mocap_Joints/"+ subject +"/MOCAP_V1_"+str(take)+".mat"
        footp_file = "./dataset/Foot_Pressure/"+ subject +"/Pressure_"+str(take)+".mat"
        save_dir = "./dataset_v1/"+ subject +"/"
        if os.path.exists(video_path) and os.path.exists(mocap_file) and os.path.exists(footp_file):
            print("Processing subject", sub_num, "take", take)
            save_frames(video_path, mocap_file, footp_file, save_dir, 
                    all_keyframes, sub_takes, crop_size, M, N, subject, take)
        else:
            print("Missing files for subject", sub_num, "take", take)
