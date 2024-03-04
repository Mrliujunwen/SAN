import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import train_transforms, get_boxes_from_mask, init_point_sampling
import json
import random
from models.auto_prompt import NuSeg
from models import auto_prompt
import argparse
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import yaml
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="dsb2018_96_NestedUNet_woDS",
                        help='model name')

    args = parser.parse_args()

    return args
def plot_examples(datax, datay, model, num_examples=6):
    fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18, 4 * num_examples))
    m = datax.shape[0]
    image_indx = np.random.randint(m)
    image_arr = model(datax[image_indx:image_indx + 1])[0].squeeze(0).detach().cpu().numpy()
    # return image_arr
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
        ax[row_num][1].imshow(np.squeeze((image_arr < 0.4)[0, :, :].astype(int)), cmap='gray')

        # ax[row_num][1].imshow(binary_image)
        ax[row_num][1].set_title("Segmented Image localization")
        ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1, 2, 0)))
        ax[row_num][2].set_title("Target image")
    plt.show()

def predict(input, target):
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # print('-' * 20)
    # for key in config.keys():
    #     print('%s: %s' % (key, str(config[key])))
    # print('-' * 20)
    #
    # cudnn.benchmark = True
    #
    # # create model
    # print("=> creating model %s" % config['arch'])
    # model = train.__dict__[config['arch']](config['num_classes'],
    #                                        config['input_channels'],
    #                                        config['deep_supervision'])
    model = auto_prompt.NuSeg(img_size=256, in_chans=3)
    model = model.cuda()
    model.load_state_dict(torch.load('saved/model_epoch26.pt'))

    model.eval()
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    # fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18, 4 * num_examples))
    m = input.shape[0]
    image_indx = np.random.randint(m)
    image_arr = model(input[image_indx:image_indx + 1])[0].squeeze(0).detach().cpu().numpy()
    return image_arr
    plot_examples(input, target, model, num_examples=3)

class TestingDataset(Dataset):

    def __init__(self, data_path, image_size=256, mode='test', requires_name=True, point_num=1, return_ori_mask=True,
                 prompt_path=None,auto_prompt=False):
        """
        Initializes a TestingDataset object.
        Args:
            data_path (str): The path to the data.
            image_size (int, optional): The size of the image. Defaults to 256.
            mode (str, optional): The mode of the dataset. Defaults to 'test'.
            requires_name (bool, optional): Indicates whether the dataset requires image names. Defaults to True.
            point_num (int, optional): The number of points to retrieve. Defaults to 1.
            return_ori_mask (bool, optional): Indicates whether to return the original mask. Defaults to True.
            prompt_path (str, optional): The path to the prompt file. Defaults to None.
        """
        self.image_size = image_size
        self.return_ori_mask = return_ori_mask
        self.prompt_path = prompt_path
        self.prompt_list = {} if prompt_path is None else json.load(open(prompt_path, "r"))
        self.requires_name = requires_name
        self.point_num = point_num
        self.auto_prompt=auto_prompt
        json_file = open(os.path.join(data_path, f'label2image_{mode}.json'), "r")
        dataset = json.load(json_file)

        self.image_paths = list(dataset.values())
        self.label_paths = list(dataset.keys())
        # testloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

    def __getitem__(self, index):
        """
        Retrieves and preprocesses an item from the dataset.
        Args:
            index (int): The index of the item to retrieve.
        Returns:
            dict: A dictionary containing the preprocessed image and associated information.
        """
        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        mask_path = self.label_paths[index]
        ori_np_mask = cv2.imread(mask_path, 0)

        if ori_np_mask.max() == 255:
            ori_np_mask = ori_np_mask / 255

        assert np.array_equal(ori_np_mask, ori_np_mask.astype(
            bool)), f"Mask should only contain binary values 0 and 1. {self.label_paths[index]}"

        h, w = ori_np_mask.shape
        ori_mask = torch.tensor(ori_np_mask).unsqueeze(0)

        transforms = train_transforms(self.image_size, h, w)
        augments = transforms(image=image, mask=ori_np_mask)
        image, mask = augments['image'], augments['mask'].to(torch.int64)

        if self.prompt_path is None:
            # 3,512,515 512,512
            if self.auto_prompt:
                im=predict(image.float().to(device).reshape(-1,3,256,256),mask.float().to(device).reshape(-1,1,256,256))
                t=im[0]


                boxes = get_boxes_from_mask(im[0])
            else:
                boxes = get_boxes_from_mask(mask)
            point_coords, point_labels = init_point_sampling(mask, self.point_num)
        else:
            prompt_key = mask_path.split('/')[-1]
            boxes = torch.as_tensor(self.prompt_list[prompt_key]["boxes"], dtype=torch.float)
            point_coords = torch.as_tensor(self.prompt_list[prompt_key]["point_coords"], dtype=torch.float)
            point_labels = torch.as_tensor(self.prompt_list[prompt_key]["point_labels"], dtype=torch.int)

        image_input["image"] = image
        image_input["label"] = mask.unsqueeze(0)
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["boxes"] = boxes
        image_input["original_size"] = (h, w)
        image_input["label_path"] = '/'.join(mask_path.split('/')[:-1])

        if self.return_ori_mask:
            image_input["ori_label"] = ori_mask

        image_name = self.label_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input

    def __len__(self):
        return len(self.label_paths)


class TrainingDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

        dataset = json.load(open(os.path.join(data_dir, f'image2label_{mode}.json'), "r"))
        # dataset = json.load(open(os.path.join(data_dir, f'split_file_1.json'), "r"))

        self.image_paths = list(dataset.keys())
        self.label_paths = list(dataset.values())

    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)

        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = random.choices(self.label_paths[index], k=self.mask_num)
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input

    def __len__(self):
        return len(self.image_paths)


def stack_dict_batched(batched_input):
    out_dict = {}
    for k, v in batched_input.items():
        if isinstance(v, list):
            out_dict[k] = v
        else:
            out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict


if __name__ == "__main__":
    train_dataset = TestingDataset("\data\histology", image_size=256, mode='test', requires_name=True, point_num=1)
    print("Dataset:", len(train_dataset))
    train_batch_sampler = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0)
    for i, batched_image in enumerate(tqdm(train_batch_sampler)):
        batched_image = stack_dict_batched(batched_image)
        print(batched_image["image"].shape, batched_image["label"].shape)

