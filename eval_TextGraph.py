import os
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from dataset import TotalText, Ctw1500Text, Icdar15Text, Mlt2017Text, TD500Text
from network.textnet import TextNet
from util.augmentation import BaseTransform
from util.config import config as cfg, update_config, print_config
from util.option import BaseOptions
from util.visualize import visualize_detection, visualize_gt
from util.misc import to_device, mkdirs, rescale_result
from util.detection_mask import TextDetector as TextDetector_mask
from util.detection_graph import TextDetector as TextDetector_graph
from eval import deal_eval_total_text, deal_eval_ctw1500, deal_eval_icdar15, \
    deal_eval_TD500, data_transfer_ICDAR, data_transfer_TD500, data_transfer_MLT2017


import multiprocessing
multiprocessing.set_start_method("spawn", force=True)


def osmkdir(out_dir):
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)


def write_to_file(contours, file_path):
    """
    :param contours: [[x1, y1], [x2, y2]... [xn, yn]]
    :param file_path: target file path
    """
    # according to total-text evaluation method, output file shoud be formatted to: y0,x0, ..... yn,xn
    with open(file_path, 'w') as f:
        for cont in contours:
            cont = np.stack([cont[:, 0], cont[:, 1]], 1)
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f.write(cont + '\n')


def inference(detector, test_loader, output_dir):

    total_time = 0.
    if cfg.exp_name != "MLT2017":
        osmkdir(output_dir)
    else:
        if not os.path.exists(output_dir):
            mkdirs(output_dir)
    for i, (image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta) in enumerate(test_loader):

        image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = to_device(
            image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)

        torch.cuda.synchronize()
        start = time.time()

        idx = 0 # test mode can only run with batch_size == 1

        # visualization
        img_show = image[idx].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

        # get detection result
        contours, output = detector.detect(image, img_show)
        tr_pred, tcl_pred = output['tr'], output['tcl']

        torch.cuda.synchronize()
        end = time.time()
        total_time += end - start
        fps = (i + 1) / total_time
        print('detect {} / {} images: {}. ({:.2f} fps)'.format(i + 1, len(test_loader), meta['image_id'][idx], fps))

        pred_vis = visualize_detection(img_show, contours, tr_pred[1], tcl_pred[1])

        gt_contour = []
        for annot, n_annot in zip(meta['annotation'][idx], meta['n_annotation'][idx]):
            if n_annot.item() > 0:
                gt_contour.append(annot[:n_annot].int().cpu().numpy())
        gt_vis = visualize_gt(img_show, gt_contour,
                              tr_mask[idx].cpu().numpy(), tcl_mask[idx, :, :, 0].cpu().numpy())
        im_vis = np.concatenate([pred_vis, gt_vis], axis=0)
        # path = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name), meta['image_id'][idx])
        # cv2.imwrite(path, im_vis)

        H, W = meta['Height'][idx].item(), meta['Width'][idx].item()
        img_show, contours = rescale_result(img_show, contours, H, W)

        # write to file
        if cfg.exp_name == "Icdar2015":
            fname = "res_" + meta['image_id'][idx].replace('jpg', 'txt')
            contours = data_transfer_ICDAR(contours)
            write_to_file(contours, os.path.join(output_dir, fname))
        elif cfg.exp_name == "MLT2017":
            path = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name),
                                meta['image_id'][idx].split("/")[-1])
            cv2.imwrite(path, im_vis)

            out_dir = os.path.join(output_dir, str(cfg.checkepoch))
            if not os.path.exists(out_dir):
                mkdirs(out_dir)
            fname = meta['image_id'][idx].split("/")[-1].replace('ts', 'res')
            fname = fname.split(".")[0] + ".txt"
            data_transfer_MLT2017(contours, os.path.join(out_dir, fname))
        elif cfg.exp_name == "TD500":
            fname = "res_img_" + meta['image_id'][idx].replace('jpg', 'txt')
            data_transfer_TD500(contours, os.path.join(output_dir, fname))

        else:
            fname = meta['image_id'][idx].replace('jpg', 'txt')
            write_to_file(contours, os.path.join(output_dir, fname))


def main(vis_dir_path):

    osmkdir(vis_dir_path)
    if cfg.exp_name == "Totaltext":
        testset = TotalText(
            data_root='data/total-text-mat',
            ignore_list=None,
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )

    elif cfg.exp_name == "Ctw1500":
        testset = Ctw1500Text(
            data_root='data/ctw1500',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.exp_name == "Icdar2015":
        testset = Icdar15Text(
            data_root='data/Icdar2015',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.exp_name == "MLT2017":
        testset = Mlt2017Text(
            data_root='data/MLT2017',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.exp_name == "TD500":
        testset = TD500Text(
            data_root='data/TD500',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    else:
        print("{} is not justify".format(cfg.exp_name))

    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    # Model
    model = TextNet(is_training=False, backbone=cfg.net)
    model_path = os.path.join(cfg.save_dir, cfg.exp_name,
                              'textgraph_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))
    model.load_model(model_path)

    # copy to cuda
    model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True

    if cfg.graph_link:
        detector = TextDetector_graph(model)
    else:
        detector = TextDetector_mask(model)

    print('Start testing TextGraph.')
    output_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    inference(detector, test_loader, output_dir)

    if cfg.exp_name == "Totaltext":
        deal_eval_total_text(debug=True)

    elif cfg.exp_name == "Ctw1500":
        deal_eval_ctw1500(debug=True)

    elif cfg.exp_name == "Icdar2015":
        deal_eval_icdar15(debug=True)
    elif cfg.exp_name == "TD500":
        deal_eval_TD500(debug=True)
    else:
        print("{} is not justify".format(cfg.exp_name))


if __name__ == "__main__":
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    vis_dir = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name))

    if not os.path.exists(vis_dir):
        mkdirs(vis_dir)
    # main
    main(vis_dir)
