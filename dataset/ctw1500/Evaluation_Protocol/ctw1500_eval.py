#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os, shutil, sys
from voc_eval_polygon import voc_eval_polygon
from collections import Counter
import numpy as np

import argparse
parser = argparse.ArgumentParser()

# basic opts
parser.add_argument('exp_name', type=str, help='Model output directory')
args = parser.parse_args()


input_dir = 'output/{}'.format(args.exp_name)
eval_result_dir = "output/Analysis/output_eval"

anno_path = 'data/ctw1500/test/test_label_curve.txt'
imagesetfile = 'data/ctw1500/test/test.txt'

outputstr = "dataset/ctw1500/Evaluation_sort/detections_text"


# score_thresh_list=[0.2, 0.3, 0.4, 0.5, 0.6, 0.62, 0.65, 0.7, 0.75, 0.8, 0.9]
score_thresh_list = [0.5]
files = os.listdir(input_dir)
files.sort()
for iscore in score_thresh_list:
    fpath = outputstr + str(iscore) + '.txt'
    with open(fpath, "w") as f1:
        for ix, filename in enumerate(files):
            imagename = filename[:-4]
            with open(os.path.join(input_dir, filename), "r") as f:
                lines = f.readlines()

            for line in lines:
                box = line.strip().split(",")
                assert (len(box) % 2 == 0), 'mismatch xy'
                out_str = "{} {}".format(str(int(imagename[:]) - 1001), 0.999)
                for i in box:
                    out_str = out_str + ' ' + str(i)
                f1.writelines(out_str + '\n')
    rec, prec, AP, FP, TP, image_ids, num_gt = voc_eval_polygon(fpath, anno_path, imagesetfile, 'text', ovthresh=0.5)
    fid_path = '{}/Eval_ctw1500_{}.txt'.format(eval_result_dir, iscore)
    F = lambda x, y: 2 * x * y * 1.0 / (x + y)

    img_dict = dict(Counter(image_ids))

    with open(fid_path, 'w') as f:
        count = 0
        for k, v in zip(img_dict.keys(), img_dict.values()):
            fp = np.sum(FP[count:count+v])
            tp = np.sum(TP[count:count+v])
            count += v
            recall = tp / float(num_gt[k])
            precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            f.write('%s :: Precision=%.4f - Recall=%.4f\n' % (str(int(k)+1001)+".txt", recall, precision))

        Recall = rec[-1]
        Precision = prec[-1]
        F_score = F(Recall, Precision)
        f.write('ALL :: AP=%.4f - Precision=%.4f - Recall=%.4f - Fscore=%.4f' % (AP, Precision, Recall, F_score))

    print('AP: {:.4f}, recall: {:.4f}, pred: {:.4f}, '
          'FM: {:.4f}\n'.format(AP, Recall, Precision, F_score))

