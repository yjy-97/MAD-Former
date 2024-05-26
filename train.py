import os
import argparse
import numpy as np
import random

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from my_dataset import MyDataSet
from model.MFor import MADFormer
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate

import warnings
warnings.filterwarnings('ignore')

def setup_seed(seed):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def main(args):
    setup_seed(3407)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path = r"/mnt/public/home/Yejiayu/dataset/AD_NC/train_4"
    #train_images_path = r"/mnt/public/home/Yejiayu/dataset/NC_MCINC/train4"
    #train_images_path = r"/mnt/public/home/Yejiayu/dataset/MCIC_MCINC/train4"
    #train_images_path = r"/mnt/public/home/Yejiayu/dataset/MCIC_MCINC/train5"
    #train_images_path = r"/mnt/public/home/Yejiayu/dataset/review/train"

    val_images_path = r"/mnt/public/home/Yejiayu/dataset/AD_NC/test_4"
    #val_images_path = r"/mnt/public/home/Yejiayu/dataset/NC_MCINC/test4"
    #val_images_path = r"/mnt/public/home/Yejiayu/dataset/MCIC_MCINC/test4"
    #val_images_path = r"/mnt/public/home/Yejiayu/dataset/MCIC_MCINC/test5"
    #val_images_path = r"/mnt/public/home/Yejiayu/dataset/review/test"

    batch_size = args.batch_size

    train_dataset = MyDataSet(train_images_path)

    val_dataset = MyDataSet(val_images_path)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = MADFormer(dataset='mf', _conv_repr=True, _pe_type="learned").to(device)

    # if args.weights != "":
    #     assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    #     weights_dict = torch.load(args.weights, map_location=device)["model"]
    #     for k in list(weights_dict.keys()):
    #         if "head" in k:
    #             del weights_dict[k]
    #     model.load_state_dict(weights_dict, strict=False)

    # if args.freeze_layers:
    # for name, para in model.named_parameters():
    # if "head" not in name:
    # para.requires_grad_(False)
    # else:
    # print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.95, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < val_acc:
            torch.save(model.state_dict(), "/mnt/public/home/Yejiayu/weights/A_N_best_model.pth")
            best_acc = val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=2)
    #0.001
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=5e-2)

    setup_seed(3407)
    parser.add_argument('--data-path', type=str,
                        default="/data/flower_photos")

    parser.add_argument('--weights', type=str,
                        default=r'',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
