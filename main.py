from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from timeit import time
import warnings
import argparse

import sys
import cv2
import numpy as np
import base64
import requests
import urllib
from urllib import parse
import json
import random
import time
from PIL import Image
from collections import Counter
import operator

import super_gradients
from super_gradients.training import models
from super_gradients.common.object_names import Models


from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

from reid import REID
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--videos', nargs='+',
                    help='List of videos', required=True)
parser.add_argument('-all', help='Combine all videos into one', default=True)
args = parser.parse_args()  # vars(parser.parse_args())


class LoadVideo:
    def __init__(self, path, img_size=(640, 640)):
        if not os.path.isfile(path):
            raise FileExistsError

        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        print('Length of {}: {:d} frames'.format(path, self.vn))

    def get_VideoLabels(self):
        return self.cap, self.frame_rate, self.vw, self.vh


def get_FrameLabels(frame):
    text_scale = max(1, frame.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(frame.shape[1] / 500.))
    return text_scale, text_thickness, line_thickness


def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness, text_scale):
    color = get_color(abs(track_id))
    cv2.rectangle(frame, (x1, y1), (x2, y2),
                  color=color, thickness=line_thickness)
    cv2.putText(
        frame, str(track_id), (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)


def write_results(filename, data_type, w_frame_id, w_track_id, w_x1, w_y1, w_x2, w_y2, w_wid, w_hgt):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{x2},{y2},{w},{h}\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'a') as f:
        line = save_format.format(
            frame=w_frame_id, id=w_track_id, x1=w_x1, y1=w_y1, x2=w_x2, y2=w_y2, w=w_wid, h=w_hgt)
        f.write(line)
    # print('save results to {}'.format(filename))


warnings.filterwarnings('ignore')


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def load_model():
    yolo_nas = models.get('yolo_nas_l',
                          pretrained_weights='coco').cuda()

    max_cosine_distance = 0.22
    nn_budget = None
    nms_max_overlap = 0.4

    model_filename = 'model_data/models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine', max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=100)

    return yolo_nas, encoder, metric, tracker


def detection(path):
    nms_max_overlap = 0.4
    yolo_nas, encoder, metric, tracker = load_model()

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print('Error opening video stram or file')

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_dir = 'videos_output/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    output_frames = []
    output_rectanger = []
    output_areas = []
    output_wh_ratio = []
    is_vis = True

    all_frames = []
    for video in args.videos:
        loadvideo = LoadVideo(video)
        video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()

        while True:
            ret, frame = video_capture.read()
            if ret is not True:
                video_capture.release()
                break
            all_frames.append(frame)

    frame_nums = len(all_frames)
    tracking_path = out_dir + 'tracking' + '.mp4'
    combined_videos = out_dir + 'all_videos' + '.mp4'

    if is_vis:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tracking_path, fourcc, frame_rate, (w, h))
        out_2 = cv2.VideoWriter(combined_videos, fourcc, frame_rate, (w, h))

        # Combine all videos
        for frame in all_frames:
            out_2.write(frame)
        out_2.release()

    # Initiate Tracking file
    filename = out_dir + '/tracking.txt'
    open(filename, 'w')

    fps = 0.0
    frame_cnt = 0
    t1 = time.time()

    track_cnt = dict()
    images_by_id = dict()
    ids_per_frame = []

    for frame in all_frames:
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb

        results = list(yolo_nas.predict(
            conf=0.5, iou=0.7)._image_prediction_lst)

        bboxes_xyxy = results[0].prediction.bboxes_xyxy.tolist()
        confidence = results[0].prediction.confidence.tolist()

        labels = results[0].prediction.labels.tolist()

        person_bboxes_xyxy = [bbox for i, bbox in enumerate(
            bboxes_xyxy) if labels[i] == 0]
        person_confidence = [conf for i, conf in enumerate(
            confidence) if labels[i] == 0]

        bboxes_xywh = []
        for bbox in person_bboxes_xyxy:
            bbox_xywh = [int(bbox[0]), int(bbox[1]), int(
                bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])]
            bboxes_xywh.append(bbox_xywh)

        bboxes_xywh = np.array(bboxes_xywh)

        features = encoder(frame, bbox_xywh)
        detections = [Detection(bbox, 1.0, feature)
                      for bbox, feature in zip(bbox_xywh, features)]
        text_scale, text_thickness, line_thickness = get_FrameLabels(frame)

        # Run non-maxima suppersion.
        boxes = np.array([d.tlwh for d in detection])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # call the tracker

        tracker.predict()
        tracker.update(detections)

        tmp_ids = []

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            area = (int(bbox[2]) - int(bbox[0])) * \
                ((int(bbox[3]) - int(bbox[1])))

            if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < h and bbox[2] < w:
                tmp_ids.append(track.track_id)

                if track.track_id not in track_cnt:
                    track_cnt[track.track_id] = [
                        [frame_cnt, int(bbox[0]), int(bbox[1]),
                         int(bbox[2]), int(bbox[3]), area]
                    ]
                    images_by_id[track.track_id] = [
                        frame[int(bbox[1]): int(bbox[3]), int(bbox[0]):int(bbox[2])]]
                else:
                    track_cnt[track.track_id].append([
                        frame_cnt,
                        int(bbox[0]),
                        int(bbox[1]),
                        int(bbox[2]),
                        int(bbox[3]),
                        area
                    ])
                    images_by_id[track.track_id].append(
                        frame[
                            int(bbox[1]): int(bbox[3]),
                            int(bbox[0]): int(bbox[2])
                        ]
                    )

            cv2_addBox(
                track.track_id,
                frame,
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
                line_thickness,
                text_thickness,
                text_scale
            )

            write_results(
                filename, 'mot', frame_cnt+1, str(track.track_id),
                int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                w, h
            )
        ids_per_frame.append(set(tmp_ids))
        
        # save a frame
        if is_vis:
            out.write(frame)
        t2 = time.time()
        
        frame_cnt += 1
        print(frame_cnt, '/', frame_nums)
    
    if is_vis:
        out.release()
    print(f'Tracking finished in {int(time.time() - t1)}')
    print('Tracked video : {}'.format(tracking_path))
    print('Combined video : {}'.format(combined_videos))
    
    reid = REID()
    threshold = 320
    exist_ids = set()
    final_fuse_id = dict()
    
    print(f'Total IDs = {len(images_by_id)}')
    feats = dict()
    for i in images_by_id:
        print(f'ID number {i} -> Number of frames {len(images_by_id[i])}')
        feats[i] = reid._features(images_by_id[i]) # reid._features(images_by_id[i][:min(len(images_by_id[i]),100)])
    
    for f in images_by_id:
        if f:
            if len(exist_ids) == 0:
                for i in f:
                    final_fuse_id[i] = [i]
                exist_ids = exist_ids or f
            
            else:
                new_ids = f - exist_ids
                for nid in new_ids:
                    dis = []
                    if len(images_by_id[nid]) < 10:
                        exist_ids.add(nid)
                        continue
                    unpickable = []
                    for i in f:
                        for key,item in final_fuse_id.items():
                            if i in item:
                                unpickable += final_fuse_id[key]
                    
                    print('exist_ids {} unpickable {}'.format(exist_ids,unpickable))

                    for oid in (exist_ids - set(unpickable) & set(final_fuse_id.keys())):
                        tmp = np.mean(reid.compute_distance(feats[nid],feats[oid]))
                        print('nid {}, oid {}, tmp {}'.format(nid, oid,tmp))
                        dis.append([oid,tmp])
                    
                    exist_ids.add(nid)
                    
                    if not dis:
                        final_fuse_id[nid] = [nid]
                        continue
                    
                    dis.sort(key= operator.itemgetter(1))
                    if dis[0][1] < threshold:
                        combined_ids = dis[0][0]
                        images_by_id[combined_ids] += images_by_id[nid]
                        final_fuse_id[combined_ids].append(nid)
                    else:
                        final_fuse_id[nid] = [nid]