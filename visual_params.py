import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def summary(model, input_size, inputs=None, batch_size=-1, dtypes=None):
    result, params_info = summary_string(
        model, input_size, inputs, batch_size, dtypes)
    print(result)

    return params_info


def summary_string(model, input_size, inputs=None, batch_size=-1, dtypes=None ):
    # if dtypes == None:
    #     dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                for o in output:
                    if hasattr(o, 'size'):
                        summary[m_key]["output_shape"] = [-1] + list(o.size())[1:]
                    else:
                        summary[m_key]["output_shape"] = 0
                # summary[m_key]["output_shape"] = [
                #     [-1] + list(o.size())[1:] for o in output
                # ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]


    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*inputs)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "-" * 100 + "\n"
    line_new = "{:>30} {:>25} {:>25} {:>15}".format(
        "Layer (type)", "Input Shape" , "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "=" * 100 + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>30} {:>25} {:>25} {:>15}".format(
            layer,
            str(summary[layer]["input_shape"]),
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    num_devices = len(str(cfg.device).split(','))
    # num_devices = 1
    total_input_size = abs(np.sum(np.prod(x) for x in input_size)
                           * batch_size * 4. / (1024 ** 2.)) / num_devices
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.)) / num_devices # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.)) / num_devices
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "=" * 100 + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "-" * 100 + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "-" * 100 + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)




"-------------------------------------------------"


import os
import gc
import time
from datetime import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler

from dataset import SynthText, TotalText, Ctw1500Text, Icdar15Text, Mlt2017Text, TD500Text , VietSceneText
from network.loss import TextLoss
from network.textnet import TextNet
from util.augmentation import Augmentation
from util.config import config as cfg, update_config, print_config
from util.misc import AverageMeter
from util.misc import mkdirs, to_device
from util.option import BaseOptions
from util.visualize import visualize_network_output
from util.summary import LogSummary
from util.shedule import FixLR


def main():

    global lr
    if cfg.exp_name == 'Totaltext':
        trainset = TotalText(
            data_root='data/total-text-mat',
            ignore_list=None,
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        # valset = TotalText(
        #     data_root='data/total-text-mat',
        #     ignore_list=None,
        #     is_training=False,
        #     transform=BaseTransform(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        # )
        valset = None

    elif cfg.exp_name == 'Synthtext':
        trainset = SynthText(
            data_root='data/SynthText',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'Ctw1500':
        trainset = Ctw1500Text(
            data_root='data/ctw1500',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'Icdar2015':
        trainset = Icdar15Text(
            data_root='data/Icdar2015',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None
    elif cfg.exp_name == 'MLT2017':
        trainset = Mlt2017Text(
            data_root='data/MLT2017',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'TD500':
        trainset = TD500Text(
            data_root='data/TD500',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'VietSceneText':
        trainset = VietSceneText(
            data_root='data/VietSceneText',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    else:
        print("dataset name is not correct")

    train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size,
                                   shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    log_dir = os.path.join(cfg.log_dir, datetime.now().strftime('%b%d_%H-%M-%S_') + cfg.exp_name)
    logger = LogSummary(log_dir)

    # Model
    model = TextNet(backbone=cfg.net, is_training=True,use_atten=cfg.attn)
    if cfg.mgpu:
        model = nn.DataParallel(model)

    model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True

    if cfg.resume:
        load_model(model, cfg.resume)

    criterion = TextLoss()

    lr = cfg.lr
    moment = cfg.momentum
    if cfg.optim == "Adam" or cfg.exp_name == 'Synthtext':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=moment)

    if cfg.exp_name == 'Synthtext':
        scheduler = FixLR(optimizer)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.90)

    print('Start training TextGraph.')

    scheduler.step()


    # register hook
    # from torchinfo import summary as torchinfo_sum
    # print(model)
    ########## Loading training model here ##########
    for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi) in enumerate(train_loader):
        shapes = list(map(tuple,[img.shape[1:],gt_roi.shape[1:]]))
        print(shapes)
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map \
            = to_device(img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)
        # torchinfo_sum(model, input_data=[img,gt_roi])
        inputs = (img,gt_roi,to_device)
 
        summary(model,shapes,inputs=inputs)

        break
    #################################################
    print('End.')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    np.random.seed(2020)
    torch.manual_seed(2020)
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    # main
    main()
