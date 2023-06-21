import os

__all__ = ["proj_root", "arg", "netChannel"]

netChannel = {"ResNet": [64, 256, 512, 1024, 2048],
              "inter": [128, 128, 128, 128, 128],
              "final": [64, 64, 64, 64, 64]}

proj_root = os.path.dirname(__file__)
dataset_root = os.path.join(proj_root, "data/dataset")

my = os.path.join(dataset_root, "my")

NJUD_NLPR_TR=os.path.join(dataset_root, "NJUD+NLPR_TR")
NJUD_NLPR_DUT_TR=os.path.join(dataset_root, "NJUD+NLPR+DUT-RGBD_TR")

DUT_RGBD_TE= os.path.join(dataset_root, "DUT-RGBD/test_data")
LFSD=os.path.join(dataset_root, "LFSD")
NJUD_TE=os.path.join(dataset_root, "NJUD/test_data")
NLPR_TE=os.path.join(dataset_root, "NLPR/test_data")
RGBD135=os.path.join(dataset_root, "RGBD135")
SIP=os.path.join(dataset_root, "SIP")
SSD=os.path.join(dataset_root, "SSD")
STERE=os.path.join(dataset_root, "STERE")
STEREO=os.path.join(dataset_root, "STEREO")
NJUD_NLPR_DUT_TE=os.path.join(dataset_root, "NJUD+NLPR+DUT-RGBD_TE")
DEPTH_TR=os.path.join(dataset_root, "DEPTH_TR")

DUT_RGB_TE= os.path.join(dataset_root, "DUT_RGB_TE")
DUT_RGB_TR= os.path.join(dataset_root, "DUT_RGB_TR")

MIRROR_TR=os.path.join(dataset_root, "MIRROR_TR")
MIRROR_TE=os.path.join(dataset_root, "MIRROR_TE")

arg = {
    #=========模型和骨干网络================#
    "model_name": "第四章最终",
    "backbone": "convnext",
    # "backbone": "inceptionnext",
    # "backbone": "PVT_v2",
    # "backbone": "ghostnet",
    # "backbone": "vgg16",
    # "backbone": "resnet50",
    # "model_name": "PVTUNet",
    #=========加载的方式以及哪一代================#
    # "load_mode":"best",
    # "load_mode":"final",
    "load_mode":"which",
    "load_epoch":17,

    #==========程序运行模式=======================#
    # "mode":"train",
    "mode":"save_pre_picture",
    # "mode":"calcullate_each_epoch_metric",
    #

    # ===========计算每一代指标的范围================#
    "start":12,
    "end":100,

    # load dataset
    "input_size": 320,
    "batch_size": 4,
    "num_workers": 20,
    # train


    #"pretrained": os.path.join(proj_root, "model/p2t_tiny.pth"),
    "pretrained": os.path.join(proj_root, "model/pvt_v2_b2_li.pth"),
    "whether_load_pre_depth": False,
    "whether_load_edge": True,

    "mid_output": False,     #tensorboard输出网络中的特征图，只显示16个通道，只在测试时输出，测试集每张都输出

    "rest_train":False,      #训练时是否周期性休眠，减少训练时电脑卡顿。
    "epoch": 100,
    "print_fre": 50,
    # optimizer
    "optimizer":"AdamW",    #选择优化器，AdamW或者SGD
    # "lr": 0.01,
    # "lr": 5e-5,
    "lr": 1e-4,
    "lr_strategy":"every_epoch",    #学习率下降策略：every_epoch每代都以固定系数衰减，another其他策略
    "decay_ratio":0.88,             #every_epoch每代都以固定系数衰减，衰减系数调整
    "momentum": 0.9,
    "weight_decay": 1e-5,
    "nesterov": False,
    "lr_decay": 0.9,
    "depth_jpg":[NJUD_TE,NLPR_TE,NJUD_NLPR_TR,my],
    # test

    # dataset
    # "tr_data_path": NJUD_NLPR_TR,
    "tr_data_path": NJUD_NLPR_DUT_TR,
    # "tr_data_path": MIRROR_TR,
    # "tr_data_path": DUT_RGB_TR,
    # "tr_data_path": DEPTH_TR,
    "valid_data_path": NJUD_NLPR_DUT_TE,
# "valid_data_path": MIRROR_TE,
    # "valid_data_path": DUT_RGB_TE,
    "te_data_list_small": {
        # "my": my,
        "NJUD": NJUD_TE,
        "NLPR":NLPR_TE,
        "DUT_RGBD":DUT_RGBD_TE},


    "te_data_list": {
        # "DUT_RGB":DUT_RGB_TE,
        # "my": my,
        # "NJUD_NLPR_DUT_TR": NJUD_NLPR_DUT_TR,
        # "NJUD_NLPR_DUT_TE": NJUD_NLPR_DUT_TE,
        "NJUD": NJUD_TE,
        "NLPR":NLPR_TE,
        "DUT_RGBD":DUT_RGBD_TE,
        "LFSD": LFSD,
        # "RGBD135": RGBD135,
        "SIP": SIP,
        # "SSD": SSD,
        "STERE": STERE
        # "STEREO": STEREO
        # "my": my
    },
# "te_data_list": {
#         "MIRROR_TE":MIRROR_TE
#     },
}
