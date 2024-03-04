from segment_anything import sam_model_registry, SamPredictor
import torch.nn as nn
import torch
import argparse
import os
from torch import optim
from torch.utils.data import DataLoader
from DataLoader import TrainingDataset, stack_dict_batched
from utils import FocalDiceloss_IoULoss, get_logger, generate_point, setting_prompt_none, save_masks
from metrics import SegMetrics
# import time
from tqdm import tqdm
import numpy as np
# import datetime
from torch.nn import functional as F
# from apex import amp
import random
from DataLoader import TestingDataset
from util2s import *
import json
from datetime import datetime
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="SAN", help="run model name")
    parser.add_argument("--epochs", type=int, default=150, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=5, help="get mask number")
    parser.add_argument("--data_path", type=str, default=r".\histology", help="train data path")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--prompt_path", type=str, default=None, help="fix prompt path")
    parser.add_argument("--boxes_prompt", type=bool, default=False, help="use boxes prompt")
    parser.add_argument("--save_pred", type=bool, default=False, help="save reslut")
    parser.add_argument("--point_num", type=int, default=1, help="point num")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="SAN\pretrain_model\san.pth", help="sam checkpoint")
    parser.add_argument("--iter_point", type=int, default=8, help="point iterations")
    parser.add_argument('--lr_scheduler', type=str, default=None, help='lr scheduler')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--use_amp", type=bool, default=False, help="use amp")
    args = parser.parse_args()
    if args.resume is not None:
        args.sam_checkpoint = None
    return args


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key == 'image' or key == 'label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=False):
    if batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
            )

    else:
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )

    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i + 1, idx])
        low_res_masks = torch.stack(low_res, 0)

    masks = F.interpolate(low_res_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False, )
    return masks, low_res_masks, iou_predictions


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion):
    train_loader = tqdm(train_loader)
    train_losses = []
    train_iter_metrics = [0] * len(args.metrics)
    for batch, batched_input in enumerate(train_loader):
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)

        if random.random() > 0.5:
            batched_input["point_coords"] = None
            flag = "boxes"
        else:
            batched_input["boxes"] = None
            flag = "point"

        for n, value in model.image_encoder.named_parameters():
            if "Adapter" in n:
                value.requires_grad = True
            else:
                # value.requires_grad = False
                value.requires_grad = True


        if args.use_amp:
            labels = batched_input["label"].half()
            image_embeddings = model.image_encoder(batched_input["image"].half())

            batch, _, _, _ = image_embeddings.shape
            image_embeddings_repeat = []
            for i in range(batch):
                image_embed = image_embeddings[i]
                image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                image_embeddings_repeat.append(image_embed)
            image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings,
                                                                       decoder_iter=False)
            loss = criterion(masks, labels, iou_predictions)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=False)

        else:
            labels = batched_input["label"]
            image_embeddings = model.image_encoder(batched_input["image"])

            batch, _, _, _ = image_embeddings.shape
            image_embeddings_repeat = []
            for i in range(batch):
                image_embed = image_embeddings[i]
                image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                image_embeddings_repeat.append(image_embed)
            image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings,
                                                                       decoder_iter=False)
            loss = criterion(masks, labels, iou_predictions)
            loss.backward(retain_graph=False)

        optimizer.step()
        optimizer.zero_grad()

        if int(batch + 1) % 50 == 0:
            print(
                f'Epoch: {epoch + 1}, Batch: {batch + 1}, first {flag} prompt: {SegMetrics(masks, labels, args.metrics)}')

        point_num = random.choice(args.point_list)
        batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
        batched_input = to_device(batched_input, args.device)

        image_embeddings = image_embeddings.detach().clone()
        for n, value in model.named_parameters():
            if "image_encoder" in n:
                # value.requires_grad = False
                value.requires_grad = True

            else:
                value.requires_grad = True
        # print('num params: {}'.format(compute_n_params(model)))

        init_mask_num = np.random.randint(1, args.iter_point - 1)
        for iter in range(args.iter_point):
            if iter == init_mask_num or iter == args.iter_point - 1:
                batched_input = setting_prompt_none(batched_input)

            if args.use_amp:
                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings,
                                                                           decoder_iter=True)
                loss = criterion(masks, labels, iou_predictions)
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
            else:
                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings,
                                                                           decoder_iter=True)
                loss = criterion(masks, labels, iou_predictions)
                loss.backward(retain_graph=True)

            optimizer.step()
            optimizer.zero_grad()

            if iter != args.iter_point - 1:
                point_num = random.choice(args.point_list)
                batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
                batched_input = to_device(batched_input, args.device)

            if int(batch + 1) % 50 == 0:
                if iter == init_mask_num or iter == args.iter_point - 1:
                    print(
                        f'Epoch: {epoch + 1}, Batch: {batch + 1}, mask prompt: {SegMetrics(masks, labels, args.metrics)}')
                else:
                    print(
                        f'Epoch: {epoch + 1}, Batch: {batch + 1}, point {point_num} prompt: {SegMetrics(masks, labels, args.metrics)}')

        if int(batch + 1) % 200 == 0:
            print(f"epoch:{epoch + 1}, iteration:{batch + 1}, loss:{loss.item()}")
            save_path = os.path.join(f"{args.work_dir}/models", args.run_name,
                                     f"epoch{epoch + 1}_batch{batch + 1}_sam.pth")
            state = {'model': model.state_dict(), 'optimizer': optimizer}
            torch.save(state, save_path)

        train_losses.append(loss.item())

        gpu_info = {}
        gpu_info['gpu_name'] = args.device
        train_loader.set_postfix(train_loss=loss.item(), gpu_info=gpu_info)

        train_batch_metrics = SegMetrics(masks, labels, args.metrics)
        train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]

    return train_losses, train_iter_metrics
















def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    if ori_h < image_size and ori_w < image_size:
        top = torch.div((image_size - ori_h), 2, rounding_mode='trunc')  # (image_size - ori_h) // 2
        left = torch.div((image_size - ori_w), 2, rounding_mode='trunc')  # (image_size - ori_w) // 2
        masks = masks[..., top: ori_h + top, left: ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None
    return masks, pad



def test_one_epoch(test_loader, model, epoch):
    Acc_list = []
    F1_list = []
    IOU_list = []
    aji_score_list = 0.0
    IoU_list = 0.0
    F1_score_list = 0.0
    acc_list = 0.0
    model =model
    test_pbar = tqdm(test_loader)
    l = len(test_loader)
    criterion = FocalDiceloss_IoULoss()
    model.eval()
    now = datetime.now()

    test_loss = []
    test_iter_metrics = [0] * len(args.metrics)
    test_metrics = {}
    prompt_dict = {}

    print(now)
    for i, batched_input in enumerate(test_pbar):
        # now = datetime.now()
        batched_input = to_device(batched_input, args.device)
        ori_labels = batched_input["ori_label"]
        original_size = batched_input["original_size"]
        labels = batched_input["label"]
        img_name = batched_input['name'][0]
        if args.prompt_path is None:
            prompt_dict[img_name] = {
                "boxes": batched_input["boxes"].squeeze(1).cpu().numpy().tolist(),
                "point_coords": batched_input["point_coords"].squeeze(1).cpu().numpy().tolist(),
                "point_labels": batched_input["point_labels"].squeeze(1).cpu().numpy().tolist()
            }

        with torch.no_grad():
            image_embeddings = model.image_encoder(batched_input["image"])

        if args.boxes_prompt:
            save_path = os.path.join(args.work_dir, args.run_name, "boxes_prompt")
            batched_input["point_coords"], batched_input["point_labels"] = None, None
            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
            points_show = None

        else:
            save_path = os.path.join(f"{args.work_dir}", args.run_name,
                                     f"iter{args.iter_point if args.iter_point > 1 else args.point_num}_prompt")
            batched_input["boxes"] = None
            point_coords, point_labels = [batched_input["point_coords"]], [batched_input["point_labels"]]

            for iter in range(args.iter_point):
                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
                if iter != args.iter_point - 1:
                    batched_input = generate_point(masks, labels, low_res_masks, batched_input, args.point_num)
                    batched_input = to_device(batched_input, args.device)
                    point_coords.append(batched_input["point_coords"])
                    point_labels.append(batched_input["point_labels"])
                    batched_input["point_coords"] = torch.concat(point_coords, dim=1)
                    batched_input["point_labels"] = torch.concat(point_labels, dim=1)

            points_show = (torch.concat(point_coords, dim=1), torch.concat(point_labels, dim=1))

        masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)
        if args.save_pred:
            save_masks(masks, save_path, img_name, args.image_size, original_size, pad,
                       batched_input.get("boxes", None), points_show)

        loss = criterion(masks, ori_labels, iou_predictions)
        test_loss.append(loss.item())
        IoU = iou(masks, ori_labels)
        F1_score = calculate_F1_score(masks, ori_labels)
        acc = calculate_acc(masks, ori_labels)
        acc_list += acc.item()
        IoU_list += IoU.item()
        F1_score_list += F1_score.item()
        Acc_list.append(acc.item())
        F1_list.append(F1_score.item())
        IOU_list.append(IoU.item())
        test_batch_metrics = SegMetrics(masks, ori_labels, args.metrics)

        test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in test_batch_metrics]
        # draw_acc(test_loss, str('now'))
        for j in range(len(args.metrics)):
            test_iter_metrics[j] += test_batch_metrics[j]
    # draw_f1(F1_list, str('training'))
    # draw_acc(Acc_list, str('training'))
    # draw_iou(IOU_list, str('training'))
    test_iter_metrics = [metric / l for metric in test_iter_metrics]
    test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}

    average_loss = np.mean(test_loss)
    if args.prompt_path is None:
        with open(os.path.join(args.work_dir, f'{args.image_size}_prompt.json'), 'w') as f:
            json.dump(prompt_dict, f, indent=2)
    print(f"Test loss: {average_loss:.4f}, metrics: {test_metrics}")
    return average_loss,test_metrics





def main(args):
    model = sam_model_registry[args.model_type](args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalDiceloss_IoULoss()

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.5)
        print('*******Use MultiStepLR')

    if args.resume is not None:
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            print(f"*******load {args.resume}")

    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        print("*******Mixed precision with Apex")
    else:
        print('*******Do not use mixed precision')

    train_dataset = TrainingDataset(args.data_path, image_size=args.image_size, mode='train', point_num=1,
                                    mask_num=args.mask_num, requires_name=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print('*******Train data:', len(train_dataset))

    test_dataset = TestingDataset(data_path=args.data_path, image_size=args.image_size, mode='test', requires_name=True,
                                  point_num=args.point_num, return_ori_mask=True, prompt_path=args.prompt_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    loggers = get_logger(
        os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.now().strftime('%Y%m%d-%H%M.log')}"))

    best_loss = 1e10
    l = len(train_loader)
    tl=len(test_loader)
    print('num params: {}'.format(compute_n_params(model)))

    for epoch in range(0, args.epochs):
        model.train()
        train_metrics = {}
        start = time.time()
        os.makedirs(os.path.join(f"{args.work_dir}/models", args.run_name), exist_ok=True)
        train_losses, train_iter_metrics = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion)
        print("test:{}".format(epoch))
        model.eval()
        test_average_loss,test_metrics=test_one_epoch(test_loader,model,epoch)
        save_path = os.path.join(args.work_dir, "models", args.run_name, f"epoch{epoch + 1}_test_sam.pth")
        state = {'model': model.float().state_dict(), 'optimizer': optimizer}
        torch.save(state, save_path)
        if args.lr_scheduler is not None:
            scheduler.step()
        # for param in sam.image_encoder.parameters():
        #     param.requires_grad = False

        train_iter_metrics = [metric / l for metric in train_iter_metrics]
        train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in
                         range(len(train_iter_metrics))}

        average_loss = np.mean(train_losses)
        lr = scheduler.get_last_lr()[0] if args.lr_scheduler is not None else args.lr
        loggers.info(f"epoch: {epoch + 1}, lr: {lr}, Train loss: {average_loss:.4f}, metrics: {train_metrics},test loss:{test_average_loss:.4f},test metrics{test_metrics}"
                     )
        loggers.info(
            f"epoch: {epoch + 1}, lr: {lr}, Train loss: {average_loss:.4f}, metrics: {train_metrics}"
            )

        # if average_loss < best_loss:
        #     best_loss = average_loss
        #     save_path = os.path.join(args.work_dir, "models", args.run_name, f"epoch{epoch + 1}_sam.pth")
        #     state = {'model': model.float().state_dict(), 'optimizer': optimizer}
        #     torch.save(state, save_path)
        #     if args.use_amp:
        #         model = model.half()

        end = time.time()
        print("Run epoch time: %.2fs" % (end - start))


if __name__ == '__main__':
    args = parse_args()
    main(args)


