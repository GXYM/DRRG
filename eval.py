import os
import cv2
import numpy as np
import subprocess
from util.config import config as cfg
from util.misc import mkdirs


def osmkdir(out_dir):
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)


def analysize_result(source_dir, fid_path, outpt_dir, name):

    bad_txt = open("{}/eval.txt".format(outpt_dir), 'w')
    all_eval = open("{}/{}/{}_eval.txt".format(cfg.output_dir, "Analysis", name), 'a+')
    sel_list = list()
    with open(fid_path) as f:
        lines = f.read().split("\n")
        for line in lines:
            line_items = line.split(" ")
            id = line_items[0]
            precision = float(line_items[2].split('=')[-1])
            recall = float(line_items[4].split('=')[-1])
            if id != "ALL" and (precision < 0.5 or recall < 0.5):
                img_path = os.path.join(source_dir, line_items[0].replace(".txt", ".jpg"))
                os.system('cp {} {}'.format(img_path, outpt_dir))
                sel_list.append((int(id.replace(".txt", "").replace("img", "").replace("_", "")), line))
            if id == "ALL":
                all_eval.write("{} {} {}\n".format(
                    outpt_dir.split('/')[-1],
                    "{}/{}/{}".format(cfg.tr_thresh, cfg.tcl_thresh, cfg.expend),
                    line))
    sel_list = sorted(sel_list, key=lambda its: its[0])
    bad_txt.write('\n'.join([its[1] for its in sel_list]))
    all_eval.close()
    bad_txt.close()


def deal_eval_total_text(debug=False):
    # compute DetEval
    eval_dir = os.path.join(cfg.output_dir, "Analysis", "output_eval")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    print('Computing DetEval in {}/{}'.format(cfg.output_dir, cfg.exp_name))
    subprocess.call(
        ['python', 'dataset/total_text/Evaluation_Protocol/Python_scripts/Deteval.py', cfg.exp_name, '--tr', '0.7',
         '--tp', '0.6'])
    subprocess.call(
        ['python', 'dataset/total_text/Evaluation_Protocol/Python_scripts/Deteval.py', cfg.exp_name, '--tr', '0.8',
         '--tp', '0.4'])

    if debug:
        source_dir = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name))
        outpt_dir_base = os.path.join(cfg.output_dir, "Analysis", "eval_view", "total_text")
        if not os.path.exists(outpt_dir_base):
            mkdirs(outpt_dir_base)

        outpt_dir1 = os.path.join(outpt_dir_base, "{}_{}_{}_{}_{}"
                                  .format(cfg.test_size[0], cfg.test_size[1], cfg.checkepoch, 0.7, 0.6))
        osmkdir(outpt_dir1)
        fid_path1 = '{}/Eval_TotalText_{}_{}.txt'.format(eval_dir, 0.7, 0.6)

        analysize_result(source_dir, fid_path1, outpt_dir1, "totalText")

        outpt_dir2 = os.path.join(outpt_dir_base, "{}_{}_{}_{}_{}"
                                  .format(cfg.test_size[0], cfg.test_size[1], cfg.checkepoch, 0.8, 0.4))
        osmkdir(outpt_dir2)
        fid_path2 = '{}/Eval_TotalText_{}_{}.txt'.format(eval_dir, 0.8, 0.4)

        analysize_result(source_dir, fid_path2, outpt_dir2, "totalText")

    print('End.')


def deal_eval_ctw1500(debug=False):
    # compute DetEval
    eval_dir = os.path.join(cfg.output_dir, "Analysis", "output_eval")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    print('Computing DetEval in {}/{}'.format(cfg.output_dir, cfg.exp_name))
    subprocess.call(['python', 'dataset/ctw1500/Evaluation_Protocol/ctw1500_eval.py', cfg.exp_name])

    if debug:
        source_dir = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name))
        outpt_dir_base = os.path.join(cfg.output_dir, "Analysis", "eval_view", "ctw1500")
        if not os.path.exists(outpt_dir_base):
            mkdirs(outpt_dir_base)

        outpt_dir = os.path.join(outpt_dir_base, "{}_{}_{}".format(cfg.test_size[0], cfg.test_size[1], cfg.checkepoch))
        osmkdir(outpt_dir)
        fid_path1 = '{}/Eval_ctw1500_{}.txt'.format(eval_dir, 0.5)

        analysize_result(source_dir, fid_path1, outpt_dir, "ctw1500")

    print('End.')


def deal_eval_icdar15(debug=False):
    # compute DetEval
    eval_dir = os.path.join(cfg.output_dir, "Analysis", "output_eval")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    input_dir = 'output/{}'.format(cfg.exp_name)
    father_path = os.path.abspath(input_dir)
    print(father_path)
    print('Computing DetEval in {}/{}'.format(cfg.output_dir, cfg.exp_name))
    subprocess.call(['sh', 'dataset/icdar15/eval.sh', father_path])

    if debug:
        source_dir = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name))
        outpt_dir_base = os.path.join(cfg.output_dir, "Analysis", "eval_view", "icdar15")
        if not os.path.exists(outpt_dir_base):
            mkdirs(outpt_dir_base)

        outpt_dir = os.path.join(outpt_dir_base, "{}_{}_{}".format(cfg.test_size[0], cfg.test_size[1], cfg.checkepoch))
        osmkdir(outpt_dir)
        fid_path1 = '{}/Eval_icdar15.txt'.format(eval_dir)

        analysize_result(source_dir, fid_path1, outpt_dir, "icdar15")

    print('End.')

    pass


def deal_eval_TD500(debug=False):
    # compute DetEval
    eval_dir = os.path.join(cfg.output_dir, "Analysis", "output_eval")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    input_dir = 'output/{}'.format(cfg.exp_name)
    father_path = os.path.abspath(input_dir)
    print(father_path)
    print('Computing DetEval in {}/{}'.format(cfg.output_dir, cfg.exp_name))
    subprocess.call(['sh', 'dataset/TD500/eval.sh', father_path])

    if debug:
        source_dir = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name))
        outpt_dir_base = os.path.join(cfg.output_dir, "Analysis", "eval_view", "TD500")
        if not os.path.exists(outpt_dir_base):
            mkdirs(outpt_dir_base)

        outpt_dir = os.path.join(outpt_dir_base, "{}_{}_{}".format(cfg.test_size[0], cfg.test_size[1], cfg.checkepoch))
        osmkdir(outpt_dir)
        fid_path1 = '{}/Eval_TD500.txt'.format(eval_dir)

        analysize_result(source_dir, fid_path1, outpt_dir, "TD500")

    print('End.')


def data_transfer_ICDAR(contours):
    cnts = list()
    for cont in contours:
        rect = cv2.minAreaRect(cont)
        points = cv2.boxPoints(rect)
        points = np.int0(points)
        # print(points.shape)
        # points = np.reshape(points, (4, 2))
        cnts.append(points)
    return cnts


def data_transfer_TD500(contours, res_file, img=None):
    with open(res_file, 'w') as f:
        for cont in contours:
            rect = cv2.minAreaRect(cont)
            points = cv2.boxPoints(rect)
            box = np.int0(points)

            cv2.drawContours(img, [box], 0, (0, 255, 0), 3)
            # cv2.imshow("lllll", img)
            # cv2.waitKey(0)

            cx, cy = rect[0]
            w_, h_ = rect[1]
            angle = rect[2]
            mid_ = 0
            if angle > 45:
                angle = 90 - angle
                mid_ = w_;
                w_ = h_;
                h_ = mid_
            elif angle < -45:
                angle = 90 + angle
                mid_ = w_;
                w_ = h_;
                h_ = mid_
            angle = angle / 180 * 3.141592653589

            x_min = int(cx - w_ / 2)
            x_max = int(cx + w_ / 2)
            y_min = int(cy - h_ / 2)
            y_max = int(cy + h_ / 2)
            f.write('{},{},{},{},{}\r\n'.format(x_min, y_min, x_max, y_max, angle))

    return img


def data_transfer_MLT2017(contours, res_file):
    with open(res_file, 'w') as f:
        for cont in contours:
            rect = cv2.minAreaRect(cont)
            points = cv2.boxPoints(rect)
            points = np.int0(points)
            p = np.reshape(points, -1)
            f.write('{},{},{},{},{},{},{},{},{}\r\n'
                    .format(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 1))



