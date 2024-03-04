import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A
from models import auto_prompt
import train
from testdataset import Dataset
import matplotlib.colors as mcolors
# from metrics import iou_score
# from utils import AverageMeter
"""
需要指定参数：--name dsb2018_96_NestedUNet_woDS
"""

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="dsb2018_96_NestedUNet_woDS",
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    # model = train.__dict__[config['arch']](config['num_classes'],
    #                                        config['input_channels'],
    #                                        config['deep_supervision'])
    model = transnuseg.TransNuSeg(img_size=512,in_chans=3)
    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('data',config['dataset'], 'data', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('saved/model.pt'))
    model.eval()

    val_transform = Compose([
        A.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('data', config['dataset'], 'data'),
        mask_dir=os.path.join('data', config['dataset'], 'label'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        # transform=val_transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # avg_meter = AverageMeter

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):


            input = input.cuda()

            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output1,output2,output3 = model(input)

            # iou = iou_score(output, target)
            # avg_meter.update(iou, input.size(0))

            output = torch.sigmoid(output1).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    # print('IoU: %.4f' % avg_meter.avg)

    plot_examples(input, target, model,num_examples=3)

    torch.cuda.empty_cache()

# def plot_examples(datax, datay, model,num_examples=6):
#     fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18,4*num_examples))
#     m = datax.shape[0]
#     for row_num in range(num_examples):
#         image_indx = np.random.randint(m)
#         # output = model(datax[image_indx:image_indx + 1])
#         # print(output)
#
#         image_arr = model(datax[image_indx:image_indx+1])[0].squeeze(0).detach().cpu().numpy()
#         # mask = np.logical_not(image_arr > 0)
#         #
#         # # 将前景和背景颜色互换
#         # image_arr[mask] = 1  # 将背景设置为1（白色）
#         # image_arr[np.logical_not(mask)] = 0  # 将前景设置为0（黑色）
#
#         ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1,2,0)))
#         ax[row_num][0].set_title("Orignal Image")
#         ax[row_num][1].imshow(np.squeeze(~(image_arr > 0)[0,:,:].astype(int)), cmap='gray')
#         # plt.imshow(np.squeeze(image_arr[0, :, :]), cmap='gray')
#         ax[row_num][1].set_title("Segmented Image localization")
#         ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1,2,0)))
#         ax[row_num][2].set_title("Target image")
#     plt.show()

#还可以

# def plot_examples(datax, datay, model, num_examples=6):
#     # # 创建一个从黑色到荧光绿的颜色映射
#     cdict = {
#         'red': ((0.0, 0.0, 0.0),  # 黑色
#                 (1.0, 1.0, 1.0)),  # 白色中的红色分量
#
#         'green': ((0.0, 0.0, 0.0),  # 黑色
#                   (1.0, 1.05, 1.05)),  # 略高于白色的绿色分量
#
#         'blue': ((0.0, 0.0, 0.0),  # 黑色
#                  (1.0, 1.0, 1.0))  # 白色中的蓝色分量
#     }
#     green_black_cmap = mcolors.LinearSegmentedColormap('GreenBlack', segmentdata=cdict)
#
#     fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18, 4 * num_examples))
#     m = datax.shape[0]
#     for row_num in range(num_examples):
#         image_indx = np.random.randint(m)
#         image_arr = model(datax[image_indx:image_indx + 1])[0].squeeze(0).detach().cpu().numpy()
#         ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1, 2, 0)))
#         ax[row_num][0].set_title("Original Image")
#         # 使用自定义的颜色映射
#         ax[row_num][1].imshow(np.squeeze(~(image_arr > 0.4)[0, :, :].astype(int)), cmap='gray')
#         ax[row_num][1].set_title("Segmented Image localization (Green Black)")
#         ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1, 2, 0)))
#         # ax[row_num][2].imshow(datay[image_indx].transpose(1, 2, 0))
#         ax[row_num][2].set_title("Target image")
#     plt.show()


# import cv2

def plot_examples(datax, datay, model, num_examples=6):
    fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18, 4 * num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = model(datax[image_indx:image_indx + 1])[0].squeeze(0).detach().cpu().numpy()

        # 二值化
        # binary_image = (image_arr > 0.4)[0, :, :].astype(int).cmap='gray'

        # 边缘检测
        # edges = cv2.Canny(binary_image, 0, 255)

        # 将边缘涂成绿色
        # binary_image_colored = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
        # binary_image_colored = np.stack([binary_image * 255] * 3, axis=-1)  # 将二值图像转换为RGB
        # binary_image_colored[edges != 0] = [0, 255, 0]  # 在边缘位置上色

        ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1, 2, 0)))
        ax[row_num][0].set_title("Original Image")
        ax[row_num][1].imshow(np.squeeze(~(image_arr > 0.4)[0, :, :].astype(int)), cmap='gray')

        # ax[row_num][1].imshow(binary_image)
        ax[row_num][1].set_title("Segmented Image localization")
        ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1, 2, 0)))
        ax[row_num][2].set_title("Target image")
    plt.show()

if __name__ == '__main__':
    main()
