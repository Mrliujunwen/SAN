# SAN: Segment Anything Network

This repository contains the implementation of the Segment Anything Network (SAN), a framework for medical image segmentation.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/SAN.git
cd SAN
```

2. Install the required dependencies:
```bash
pip install torch torchvision opencv-python matplotlib numpy tqdm
```

## Model Preparation

Download the pretrained SAN model and place it in the correct directory:
```bash
mkdir -p SAN/pretrain_model
# Download the pretrained model to SAN/pretrain_model/san.pth
```

## Usage

### Training

To train the SAN model on your dataset:

```bash
python train.py --work_dir workdir \
                --run_name SAN \
                --data_path ./histology \
                --image_size 256 \
                --model_type vit_b \
                --sam_checkpoint SAN/pretrain_model/san.pth \
                --encoder_adapter True \
                --iter_point 8 \
                --multimask True
```

Key parameters:
- `--work_dir`: Directory to save results
- `--run_name`: Experiment name
- `--data_path`: Path to dataset
- `--image_size`: Image size for training
- `--model_type`: Model type (vit_b, vit_l, vit_h)
- `--sam_checkpoint`: Path to pretrained model
- `--encoder_adapter`: Whether to use adapter
- `--iter_point`: Point iterations for interactive segmentation

### Testing

To test the SAN model:

```bash
python test.py --work_dir workdir \
               --run_name SAN \
               --data_path ./histology \
               --image_size 256 \
               --model_type vit_b \
               --sam_checkpoint SAN/pretrain_model/san.pth \
               --encoder_adapter True \
               --boxes_prompt True \
               --point_num 3 \
               --save_pred True
```

## Project Structure

- `segment_anything/`: Core SAN model implementation
- `train.py`: Training script 
- `test.py`: Testing script
- `DataLoader.py`: Data loading utilities
- `utils.py`, `util2s.py`: Utility functions
- `metrics.py`: Evaluation metrics

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@ARTICLE{10822028,
  author={Authors},
  journal={IEEE Transactions on Medical Imaging},
  title={SAN: Segment Anything Network for Medical Image Segmentation},
  year={2024},
  volume={},
  number={},
  pages={},
  doi={10.1109/TMI.2024.10822028}
}
```

Paper link: [SAN: Segment Anything Network for Medical Image Segmentation](https://ieeexplore.ieee.org/document/10822028)

## License

This project is licensed under the terms of the LICENSE file included in this repository. 