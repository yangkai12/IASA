# -*- coding: utf-8 -*-
# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os, sys
from glob import glob

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import traceback


from pytracking.evaluation import Tracker
from pytracking.utils.convert_vot_anno_to_rect import convert_vot_anno_to_rect
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from toolkit.pysot.utils.bbox import get_axis_aligned_bbox
from pytracking.evaluation.environment import env_settings

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--tracker_name', default='trdimp', type=str, help='Name of tracking method.')
parser.add_argument('--tracker_param', default='trdimp', type=str, help='Name of parameter file.')
parser.add_argument('--dataset', default='NFS30', type=str, help='datasets')
parser.add_argument('--checkpoint', default='', type=str, help='checkpoint of models to eval')
parser.add_argument('--video', default='', type=str, help='eval one special video')
parser.add_argument('--tracker_appendix', type=str, default='', help='the appendix of the tracker name')
parser.add_argument('--vis', action='store_true', help='whether visualzie result')
parser.add_argument('--start', default=1, type=int, help='the start index of the testing checkpoints')
parser.add_argument('--end', default=1, type=int, help='the end index of the testing checkpoints')
parser.add_argument('--times', default=200, type=int, help='the name of the tracker')
args = parser.parse_args()

torch.set_num_threads(1)

env = env_settings()

def main(tracker_idx):#
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join('/data/study/code/pysot-master01/testing_dataset', args.dataset)

    # build tracker
    tracker_generator = Tracker(args.tracker_name, args.tracker_param)
    params = tracker_generator.get_parameters()
    tracker = tracker_generator.tracker_class(params)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    total_lost = 0
    model_name = str(args.tracker_param)
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):

                #if len(gt_bbox) == 4:
                #    gt_bbox = [gt_bbox[0], gt_bbox[1],
                #       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                #       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                #       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    #cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    #gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    #tracker.initx1, y1, w, h]
                    gt_bbox_ = convert_vot_anno_to_rect(gt_bbox, 'preserve_area')
                    info = {'init_bbox': gt_bbox_}
                    tracker.initialize(img, info)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    #tracker.tracknp.array)
                    outputs = tracker.track(img)
                    pred_bbox = outputs['target_bbox']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)

                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1

                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx >= frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                  (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join(env.results_path, args.tracker_name, args.tracker_param,
                                args.dataset, tracker_idx, 'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(tracker_idx, total_lost))

    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    info = {'init_bbox': gt_bbox_}
                    tracker.initialize(img, info)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['target_bbox']
                    pred_bboxes.append(pred_bbox)
                    #scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('./pytracking/tracking_results', args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('./pytracking/tracking_results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join('./pytracking/tracking_results', args.dataset, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    num_tries = 3
    for i in range(num_tries):
        try:
            for i in range(args.times):
                tracker_idx = '{:s}{:d}'.format(args.tracker_appendix, i)
                main(tracker_idx)
                a = 1
        except:
            epoch = 1
            print('Training crashed at epoch {}'.format(epoch))
            print(traceback.format_exc())
            print('Restarting training from last epoch ...')
