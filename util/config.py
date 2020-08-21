from easydict import EasyDict
import torch
import os

config = EasyDict()

config.gpu = "2"

# dataloader jobs number
config.num_workers = 0

# batch_size
config.batch_size = 2

# training epoch number
config.max_epoch = 200

config.start_epoch = 0

# learning rate
config.lr = 1e-4

# using GPU
config.cuda = True

config.k_at_hop1 = 10

config.output_dir = 'output'

config.input_size = 640

# max polygon per image
config.max_annotation = 200

# max polygon per image
# synText, total-text:600; CTW1500: 1200; icdar: ; MLT: ; TD500: .
config.max_roi = 600

# max point per polygon
config.max_points = 20

# use hard examples (annotated as '#')
config.use_hard = True

# demo tr threshold
config.tr_thresh = 0.8

# demo tcl threshold
config.tcl_thresh = 0.5

# expand ratio in post processing
config.expend = -0.05 #0.15

# k-n graph
config.k_at_hop = [8, 8]

# unn connect
config.active_connection = 3

config.graph_link = False  ### for Total-text icdar15 graph_link = False; forTD500 and CTW1500, graph_link = True
config.link_thresh = 0.85 #0.9

def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v
    # print(config.gpu)
    config.device = torch.device('cuda') if config.cuda else torch.device('cpu')


def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')
