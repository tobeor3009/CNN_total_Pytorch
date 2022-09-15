import sys
import matplotlib.pyplot as plt
import time
import numpy as np
import argparse
import math
import os
from glob import glob
import random
import json
from custom.utils import *

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm.auto import tqdm

# PyTorch model and training necessities
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam


# Image datasets and image manipulation
import torchvision
from torchvision import transforms, utils
from custom.utils import *
from custom.torch_utils import *
from custom.augmentations import *

# for distributed
from torch.utils.data.distributed import DistributedSampler
from distributed import (
    get_rank,
    synchronize,
)

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Usage1
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 patient_info_predictor.py
 
"""


class AxialScan(torch.utils.data.Dataset):
    def __init__(self,
                 split='train',
                 transform=None,
                 n_slices=32,
                 reverse=True,
                 label_dim=2,
                 ):
        self.split = split
        self.transform = transform
        self.n_slices = n_slices
        self.reverse = reverse
        self.label_dim = label_dim

        # self.dataset = [query_dir for query_dir in glob(os.path.join("../Dataset/Asan_Brain_CT/train/dicom/*/enrolled") if os.path.isdir(query_dir))]
        self.dataset = load_obj(
            "../Dataset/Asan_Brain_CT/train/png/normal/dataset.pkl")
        # split data into train : valid : test = 80 : 10 : 10
        n_patient = len(self.dataset)
        n_train = int(n_patient * 0.8)
        n_valid = int(n_patient * 0.1)
        patient_split = {"train": self.dataset[:n_train],
                         "valid": self.dataset[n_train: n_train + n_valid],
                         "test": self.dataset[n_train + n_valid:]}
        self.dataset = patient_split[split]

        self.age_dataset = {}
        if self.split == "train":
            for data in self.dataset:
                age = data["label"][0]
                if age not in self.age_dataset:
                    self.age_dataset[age] = []
                self.age_dataset[age].append(data)
            self.ages = list(self.age_dataset.keys())

        if get_rank() == 0:
            print(f"INFO.{len(self.dataset)} patients for {split}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.split == "train":
            age = random.choice(self.ages)
            data = random.choice(self.age_dataset[age])
        else:
            data = self.dataset[idx]
        query_dir, label = data["query_dir"], data["label"]

        imgs = sorted(glob(os.path.join(query_dir, "*.png")),
                      reverse=self.reverse)
        imgs = [Image.open(img) for img in imgs]
        if len(imgs) > self.n_slices:
            start_idx = random.randint(0, len(imgs) - self.n_slices)
            end_idx = start_idx + self.n_slices
            imgs = imgs[start_idx: end_idx]
        else:
            imgs = (self.n_slices - len(imgs)) * [imgs[0]] + imgs

        if self.transform:
            imgs = [self.transform(img) for img in imgs]

        imgs = torch.stack(imgs).float()
        label = torch.FloatTensor(label).float()

        return imgs, label


#---------------------loss fn----------------------#


class loss_fn(torch.nn.Module):
    def __init__(self, age_start=18, age_end=100):
        super(loss_fn, self).__init__()
        self.age_start = age_start
        self.age_end = age_end

        self.gender_loss_fn = nn.BCEWithLogitsLoss().cuda()

    def age_loss_fn(self, logit_age, y_age):
        age_arange = torch.arange(
            start=self.age_start, end=self.age_end, dtype=torch.FloatTensor).cuda() / 100
        age_arange = age_arange.unsqueeze(0).repeat((logit_age.size(0), 1))

        mean_pred = torch.einsum('bs, bs->b', logit_age, age_arange)
        var_pred = torch.einsum('bs, bs->b', logit_age,
                                (age_arange - mean_pred.unsqueeze(1)) ** 2)
        mean_y = y_age
        # var_y  = 1

        kl_loss = 0.5 * (-torch.log(var_pred) + var_pred +
                         (mean_pred - mean_y)**2 - 1.).mean()
        return kl_loss

    def forward(self, logit, y):
        logit_age = logit[..., 0:-1]
        y_age = y[:, 0]
        logit_gender = logit[..., -1]
        y_gender = y[:, -1]

        logit_age = torch.softmax(logit_age, dim=1)

        age_loss = self.age_loss_fn(logit_age, y_age)
        gender_loss = self.gender_loss_fn(logit_gender, y_gender)

        return age_loss * 0.5 + gender_loss * 0.5

#--------------------------------------------------#


def set_model(args):
    from model import TransformerClassifier
    model = TransformerClassifier(size=args.size,
                                  latent_model="transformer",
                                  n_latent_layer=args.n_latent_layer,
                                  label_dim=args.label_dim)

    if get_rank() == 0:
        print(model)
    return model.to(device), args


def train(args):
    args.n_gpu = int(os.environ["WORLD_SIZE"]
                     ) if "WORLD_SIZE" in os.environ else 1
    args.is_distributed = args.n_gpu > 1
    if args.is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://")
        synchronize()
        if get_rank() == 0:
            print(f"INFO.Distributed Launch with {args.n_gpu} GPUs!!!")

    model, args = set_model(args)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt = {}
    if args.ckpt:
        if get_rank() == 0:
            print(f"INFO.loading model check-point from {args.ckpt}")

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])

    if args.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False)

    phases = ["train", "valid"]

    my_transforms = {
        "train": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(args.size),
                transforms.RandomErasing(value=0, p=0.25),
                transforms.RandomAutocontrast(p=0.25),
                transforms.RandomApply(torch.nn.ModuleList(
                    [transforms.GaussianBlur(3, sigma=(0.1, 2.0))]), p=0.25),
                # set img range [-1,1]
                transforms.Normalize(
                    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(args.size),
                # set img range [-1,1]
                transforms.Normalize(
                    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    }

    real_dataset = {phase: AxialScan(split=phase,
                                     label_dim=args.label_dim,
                                     transform=my_transforms[phase]) for phase in phases}
    real_dataloader = {phase: DataLoader(real_dataset[phase],
                                         batch_size=args.batch_size,
                                         num_workers=0,
                                         sampler=data_sampler(
                                             real_dataset[phase], shuffle=True, distributed=args.is_distributed),
                                         drop_last=True,
                                         pin_memory=True) for phase in phases}

    "set dir"
    if get_rank() == 0:
        args.save_dir = os.path.join("checkpoint", "patient_info_predictor",
                                     f"{args.model}_{args.n_latent_layer}layered_transformer")
        ckpt_dir = os.path.join(args.save_dir, "ckpt")
        mkdirs(ckpt_dir)

        tensorboard_dir = os.path.join(args.save_dir, "tensorboard")
        mkdirs(tensorboard_dir)
        writer = SummaryWriter(tensorboard_dir)
        # tensorboard --logdir=<tensorboard_dir> --host=0.0.0.0 --port=8888
        # http://localhost:8888/

        dataiter = iter(real_dataloader["train"])
        scan, label = dataiter.next()
        _, _, c, w, h = scan.shape
        images = scan.view(-1, c, w, h)
        grid = torchvision.utils.make_grid(images)
        writer.add_image("images", grid)
        writer.add_graph(model, scan)
        writer.flush()

    criterion_age = nn.L1Loss(reduction='sum')
    criterion_sex = nn.BCELoss(reduction='sum')

    start_epoch = ckpt["epoch"] + 1 if "epoch" in ckpt else 0
    losses = ckpt["losses"] if "losses" in ckpt else {}
    if get_rank() == 0:
        print("INFO. start training...")
    for epoch in range(start_epoch, args.epochs):
        losses[epoch] = {}
        for phase in phases:
            if phase == "train":
                model.train()
            else:
                model.eval()

            total_patients = len(real_dataset[phase])
            n_patients = 0
            sex_correct = 0

            loss = age_loss = sex_loss = torch.tensor([0.], device=device)

            losses[epoch][phase] = {"age": [], "sex": []}
            if get_rank():
                pbar = enumerate(real_dataloader[phase])
            else:
                pbar = tqdm(
                    enumerate(real_dataloader[phase]), dynamic_ncols=True, smoothing=0.01)
            for step, (scan, label) in pbar:
                scan = scan.to(device)
                label = label.to(device)

                n_patients += scan.size(0)

                "forward"
                with torch.set_grad_enabled(phase == "train"):
                    pred = model(random_aug_g(scan)
                                 if phase == "train" else scan)

                    age, sex = label[:, 0], label[:, 1]
                    pred_age, pred_sex = pred[:, 0], pred[:, 1]

                    age_loss = criterion_age(pred_age, age)
                    sex_loss = criterion_sex(pred_sex, sex)

                    loss = age_loss + sex_loss

                    if phase == "train":
                        "backward + optimize only if in training phase"
                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                if get_rank() == 0:
                    losses[epoch][phase]["age"].append(age_loss.item())
                    losses[epoch][phase]["sex"].append(sex_loss.item())

                    running_age_loss = np.array(
                        losses[epoch][phase]["age"])[-1000:].mean()
                    running_sex_loss = np.array(
                        losses[epoch][phase]["sex"])[-1000:].mean()

                    sex_correct += (sex == (pred_sex > 0.5)).sum()
                    sex_acc = sex_correct / n_patients

                    writer.add_scalar("age_loss", age_loss.item(
                    ), epoch * len(real_dataloader[phase]) + step)
                    writer.add_scalar("sex_loss", sex_loss.item(
                    ), epoch * len(real_dataloader[phase]) + step)
                    writer.add_scalar(
                        "sex_acc", sex_acc, epoch * len(real_dataloader[phase]) + step)

                    description = f"[{phase}: {epoch}/{args.epochs}]"
                    description += f"progress = {(n_patients / total_patients) * 100 * args.n_gpu:.1f}%; "
                    description += f"|age - age_pred| = {100 * running_age_loss:.4f}; "
                    description += f"sex acc = {sex_acc:.4f} ({running_sex_loss :.4f}); "
                    description += f"lr = {optim.param_groups[0]['lr']:.7f};"
                    pbar.set_description(description)

            if phase == "valid":
                if get_rank() == 0:
                    model_module = model.module if args.is_distributed else model
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model_module.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "losses": losses,
                        "args": args,
                    }, os.path.join(ckpt_dir, f"{str(epoch).zfill(4)}.pth"))

        if get_rank() == 0:
            print("-" * 50)
    return 0


@torch.no_grad()
def test(args):
    # CUDA_VISIBLE_DEVICES=0,1 python3 patient_info_predictor.py --batch_size 2 --ckpt checkpoint/patient_info_predictor/TransformerClassifier_lambda\(0.9\,0.1\)/ckpt/0015.pth

    model, args = set_model(args)
    print(f"INFO.loading model check-point from {args.ckpt}")
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    args.n_gpu = torch.cuda.device_count()
    args.is_distributed = args.n_gpu > 1
    if args.is_distributed:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    my_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(args.size),
            # set img range [-1,1]
            transforms.Normalize(
                [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    real_dataset = AxialScan(split="test",
                             label_dim=args.label_dim,
                             transform=my_transforms)
    real_dataloader = DataLoader(real_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 drop_last=True,
                                 pin_memory=True)

    n = 0
    total_patients = len(real_dataset)
    pbar = tqdm(enumerate(real_dataloader), dynamic_ncols=True, smoothing=0.01)
    for step, (scan, label) in pbar:
        scan = scan.to(device)
        label = label.to(device)
        pred = model(scan)

        if step == 0:
            labels = label
            preds = pred
        else:
            labels = torch.cat([labels, label], 0)
            preds = torch.cat([preds, pred], 0)

        n += scan.size(0)
        sex_accuracy = (labels[:, 1] == (preds[:, 1] > 0.5)).sum() / n
        age_diff = 100 * F.l1_loss(labels[:, 0], preds[:, 0], reduction='mean')
        age_std = (100 * (labels[:, 0] - preds[:, 0])).std()
        progress = 100 * n / total_patients
        description = f"INFO.progress = {progress:.2f}% sex_acc = {sex_accuracy * 100:.2f}; |age - pred_age| = {age_diff:.2f} +- {age_std:.2f}"
        pbar.set_description(description)

    labels = labels.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    save_dir = os.path.join("checkpoint", "patient_info_predictor",
                            f"{args.model}_lambda({args.lambda_age},{args.lambda_sex})")
    save_obj({"label": labels, "pred": preds},
             os.path.join(save_dir, "test_result.pkl"))

    return 0


@torch.no_grad()
def anomaly_detection(args):
    # CUDA_VISIBLE_DEVICES=0,1 python3 patient_info_predictor.py --batch_size 2 --ckpt checkpoint/patient_info_predictor/TransformerClassifier_lambda\(0.9\,0.1\)/ckpt/0015.pth

    class BrainCT(torch.utils.data.Dataset):
        def __init__(self, transform, reverse=True, n_slices=32):
            self.n_slices = 32
            self.transform = transform
            self.reverse = reverse
            self.n_slices = n_slices
            DATASET_VERSION = "20220213"
            severities = ["Immediate", "Urgent",
                          "Indeterminate", "Benign", "Normal"]
            split = "internal_validation"
            self.query_save_dirs = []
            self.severity = []
            for severity in severities:
                query_dirs = glob(
                    f"../Dataset/Asan_Brain_CT/test/{split}/{DATASET_VERSION}/{severity}/*")
                self.query_save_dirs += query_dirs
                self.severity += [severity] * len(query_dirs)

        def __len__(self):
            return len(self.query_save_dirs)

        def __getitem__(self, idx):
            query_save_dir = self.query_save_dirs[idx]
            severity = self.severity[idx]
            imgs = sorted(glob(os.path.join(query_save_dir, 'png',
                          '*.png')), reverse=self.reverse)[:self.n_slices]
            imgs = [Image.open(img) for img in imgs]
            if len(imgs) > self.n_slices:
                start_idx = random.randint(0, len(imgs) - self.n_slices)
                end_idx = start_idx + self.n_slices
                imgs = imgs[start_idx: end_idx]
            else:
                imgs = (self.n_slices - len(imgs)) * [imgs[0]] + imgs

            if self.transform:
                imgs = [self.transform(img) for img in imgs]
            dcm = glob(os.path.join(query_save_dir, '2', '*.dcm'))[0]
            _, age, sex = get_patient_info(dcm)
            age /= 100.
            sex = 1. if sex == "M" else 0.
            label = torch.FloatTensor([age, sex])

            data = {"query_save_dir": query_save_dir,
                    "severity": severity,
                    "scan": torch.stack(imgs).float(),
                    "label": label}
            return data

    model, args = set_model(args)
    print(f"INFO.loading model check-point from {args.ckpt}")
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    args.n_gpu = torch.cuda.device_count()
    args.is_distributed = args.n_gpu > 1
    if args.is_distributed:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    my_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(args.size),
            # set img range [-1,1]
            transforms.Normalize(
                [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    brainct_dataset = BrainCT(my_transforms, reverse=True)
    brainct_dataloader = DataLoader(brainct_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    drop_last=False,
                                    pin_memory=True)

    severities = ["Immediate", "Urgent", "Indeterminate", "Benign", "Normal"]
    result = {severity: {"label": [], "pred": []} for severity in severities}

    pbar = tqdm(brainct_dataloader)
    for data in pbar:
        scan = data["scan"].to(device, non_blocking=True)
        label = data["label"].to(device)
        _, b, c, h, w = scan.shape
        pred = model(scan)

        severity = data["severity"][0]
        if result[severity]["label"] == []:
            result[severity]["label"] = label
            result[severity]["pred"] = pred
        else:
            result[severity]["label"] = torch.cat(
                [result[severity]["label"], label], 0)
            result[severity]["pred"] = torch.cat(
                [result[severity]["pred"], pred], 0)

    for severity in severities:
        labels = result[severity]["label"]
        preds = result[severity]["pred"]
        sex_accuracy = (labels[:, 1] == (
            preds[:, 1] > 0.5)).sum() / labels.size(0)
        age_diff = 100 * torch.mean(preds[:, 0] - labels[:, 0])
        age_l1_diff = 100 * \
            F.l1_loss(labels[:, 0], preds[:, 0], reduction='mean')
        age_std = (100 * (labels[:, 0] - preds[:, 0])).std()
        description = f"[{severity}] sex_acc = {sex_accuracy * 100:.2f}; (pred_age - age) = {age_diff:.2f}; |age - pred_age| = {age_l1_diff:.2f} +- {age_std:.2f}"
        print(description)

        result[severity]["label"] = labels.detach().cpu().numpy()
        result[severity]["pred"] = preds.detach().cpu().numpy()

    fname = os.path.join("checkpoint", "patient_info_predictor",
                         f"{args.model}_lambda({args.lambda_age},{args.lambda_sex})", "anomaly_detection_result.pkl")
    save_obj(result, fname)


if __name__ == "__main__":
    parser = argparse.augmentParser(
        description='patient info', formatter_class=argparse.augmentDefaultsHelpFormatter)
    parser.add_augment('--batch_size', type=int, default=1)  # per GPU
    parser.add_augment('--epochs', type=int, default=100)
    parser.add_augment('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_augment('--ckpt', type=str, default="")
    parser.add_augment('--size', type=int, default=512)
    parser.add_augment('--label_dim', type=int, default=2)
    parser.add_augment('--n_slices', type=int, default=32)
    parser.add_augment("--n_latent_layer", type=int, default=6)

    parser.add_augment('--train', action="store_true")
    parser.add_augment('--test', action="store_true")

    parser.add_augment("--local_rank", type=int, default=0,
                       help="local rank for distributed training")

    args = parser.parse_args()

    # anomaly_detection(args)
    # test(args)
    train(args)
