#!/usr/bin/env python3
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        # YOLOX-M architecture
        self.depth = 0.67
        self.width = 0.75

        # Dataset
        self.data_dir = "/home/nishchay/datasets/traffic_nature_13_yolox"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.num_classes = 13

        # Controlled benchmark
        self.input_size = (640, 640)
        self.test_size = (640, 640)

        # Disable YOLOX random multi-scale so nominal input
        # remains fixed at 640x640 like RT-DETR experiment.
        self.multiscale_range = 0

        # Same epoch budget as RT-DETRv2-S
        self.max_epoch = 120

        # YOLOX-native optimizer / augmentation recipe
        self.warmup_epochs = 5
        self.no_aug_epochs = 15

        self.data_num_workers = 2
        self.eval_interval = 1
        self.print_interval = 100

        # Avoid saving 120 huge history checkpoints.
        # latest + best are sufficient.
        self.save_history_ckpt = False

        self.seed = 42

        self.exp_name = "yolox_m_traffic_nature_13"
