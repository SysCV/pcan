# Getting Started
This page provides basic tutorials about the usage of PCAN. For installation instructions, please see [INSTALL.md](INSTALL.md).

## Prepare Datasets

#### Download BDD100K
We present an example based on [BDD100K](https://bdd100k.com/) dataset. Please first download the images and annotations from the [official website](https://bdd-data.berkeley.edu/). We use `detection` set, `tracking` set, `instance seg` set and `tracking & seg` set  for training, and validate our method on `tracking & seg` set.
For more details about the dataset, please refer to the [offial documentation](https://doc.bdd100k.com/download.html).

On the offical download page, the required data and annotations are

- `detection` set images: `Images (100k)`
- `detection` set annotations: `Detection 2020 Labels (det_20_labels)`
- `tracking` set images: `MOT 2020 Data (images20-track-*.zip)`
- `tracking` set annotations: `MOT 2020 Labels (box_track_20_labels)`
- `instance seg` set images: `Images (10k)`
- `instance seg` set annotations: `Instance segmentation Labels (ins_seg_labels)`
- `tracking & seg` set images: `MOTS 2020 Data (seg_track_20_images)` 
- `tracking & seg` set annotations: `MOTS 2020 Labels (seg_track_20_labels)` 


#### Convert annotations
we also provide [pre-converted coco format jsons](https://hkustconnect-my.sharepoint.com/:u:/g/personal/lkeab_connect_ust_hk/EWhaLPcWjQtKv9-X1dWsKwgBXdQWwP7uCpRQGUPH9uklLw?e=tlY1yz) for downloading if you want to skip the following transformation process to COCO-style.

##### BDD100K annotation transformation
To organize the annotations for training and inference, we implement a [dataset API](../pcan/datasets/parsers/coco_video_parser.py) that is similiar to COCO-style.

After downloaded the annotations, please transform the offical annotation files to CocoVID style as follows. 

First, uncompress the downloaded annotation file and you will obtain a folder named `bdd100k`.

Install repo [BDD100K data api](https://github.com/bdd100k/bdd100k).

To convert the detection set, you can do as
```bash
mkdir data/bdd/labels/det_20
python -m bdd100k.label.to_coco -m det -i data/bdd/labels/det_20/det_${SET_NAME}.json -o data/bdd/labels/det_20/det_${SET_NAME}_cocofmt.json
```

To convert the `tracking` set, you can do as
```bash
mkdir data/bdd/labels/box_track_20
python -m bdd100k.label.to_coco -m box_track -i bdd100k/labels/box_track_20/${SET_NAME} -o data/bdd/labels/box_track_20/box_track_${SET_NAME}_cocofmt.json
```

For `instance segmentation` and `segmentation tracking`, converting from “JOSN + Bitmasks” and from “Bitmask” are both supported. Use this command:
```bash
python3 -m bdd100k.label.to_coco -m ins_seg|seg_track -i ${in_path} -o ${out_path} -mb ${mask_base}
```

Some conversion examples:
```bash
python -m bdd100k.label.to_coco -m ins_seg -i data/bdd/labels/ins_seg/polygons/ins_seg_train.json -o data/bdd/labels/ins_seg/polygons/ins_seg_train_cocoformat.json -mb data/bdd/labels/ins_seg/bitmasks/train

python -m bdd100k.label.to_coco -m seg_track -i data/bdd/labels/seg_track_20/polygons/train -o data/bdd/labels/seg_track_20/seg_track_train_cocoformat.json -mb data/bdd/labels/seg_track_20/bitmasks/train
```

The `${SET_NAME}` here can be one of ['train', 'val'].

#### Symlink the data

It is recommended to symlink the dataset root to `$PCAN/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.
Our folder structure follows

```
PCAN
├── tools
├── configs
├── pcan
├── ckpts
├── scripts
├── docs
├── data
│   ├── bdd
│   │   ├── images 
│   │   │   ├── 100k 
|   |   |   |   |── train
|   |   |   |   |── val
|   |   |   |   |── test
│   │   │   ├── 10k 
|   |   |   |   |── train
|   |   |   |   |── val
|   |   |   |   |── test
│   │   │   ├── track 
|   |   |   |   |── train
|   |   |   |   |── val
|   |   |   |   |── test
│   │   │   ├── seg_track_20 
|   |   |   |   |── train
|   |   |   |   |── val
|   |   |   |   |── test
│   │   ├── labels 
│   │   │   ├── box_track_20
│   │   │   ├── seg_track_20
│   │   │   ├── det_20
│   │   │   ├── ins_seg

```

## Run PCAN
This codebase is inherited from [mmdetection](https://github.com/open-mmlab/mmdetection).
You can refer to the [offical instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/getting_started.md).
You can also refer to the short instructions below.
We provide config files in [configs](../configs).

### Train a model

Download the intial [model weights](https://hkustconnect-my.sharepoint.com/:u:/g/personal/lkeab_connect_ust_hk/EYaf3x8hV6BGiYH3RpNVhTAB1H_OefsfjkNendSEfU_zHg?e=EoaglR) from BDD100k MOT tracking set and put it under `ckpts` folder. 

#### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

If you want to specify the working directory in the command, you can add an argument `--work-dir ${YOUR_WORK_DIR}`.

#### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```
or refer to the following script:
```shell
bash scripts/train_pcan.sh
```

Optional arguments are:

- `--no-validate` (**not suggested**): By default, the codebase will perform evaluation at every k (default value is 1, which can be modified like [this](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py#L174)) epochs during the training. To disable this behavior, use `--no-validate`.
- `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--cfg-options 'Key=value'`: Overide some settings in the used config.

**Note**:

- `resume-from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
- For more clear usage, the original `load-from` is deprecated and you can use `--cfg-options 'load_from="path/to/you/model"'` instead. It only loads the model weights and the training epoch starts from 0 which is usually used for finetuning.


#### Launch multiple jobs on a single machine

If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
you need to specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs, you can set the port in commands.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

### Test a Model with COCO-format

Note that, in this repo, the evaluation metrics are computed with COCO-format.
But to report the results on BDD100K, evaluating with BDD100K-format is required.

- single GPU
- single node multiple GPU
- multiple node

Trained models for testing

- [PCAN pretrained model - ResNet50](https://hkustconnect-my.sharepoint.com/:u:/g/personal/lkeab_connect_ust_hk/EVdSxsVuKlFDg5I77VsFr4UB_KJQY4Dd_5ZUMJ6gG9A2Hw?e=IhqrfY)

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show] [--cfg-options]
```

```shell
# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--cfg-options]
```

or refer to the following script:
```shell
bash scripts/test_pcan.sh
```

Optional arguments:
- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.
- `EVAL_METRICS`: Items to be evaluated on the results. Allowed values depend on the dataset, e.g., `bbox`, `track`.
- `--cfg-options`: If specified, some setting in the used config will be overridden.

#### Visualization script
```shell
# single-gpu testing with visualization
bash scripts/vis_pcan.sh
```

### Conversion to the BDD100K/Scalabel format and Evalution

We provide [scripts](https://github.com/SysCV/pcan/blob/main/scripts/convert_to_bdd.sh) to convert the output prediction into BDD100K format jsons and masks,
which can be submitted to [BDD100K benchmark server](https://eval.ai/web/challenges/challenge-page/1295/leaderboard/3268) to get the final performance for `test set`. You can also evaluate `val set` offline using [script](https://github.com/SysCV/pcan/blob/main/scripts/eval_bdd_submit.sh) for final performance.


```shell
python tools/to_bdd100k.py ${CONFIG_FILE} [--res ${RESULT_FILE}] [--task ${EVAL_METRICS}] [--bdd-dir ${BDD_OUTPUT_DIR} --nproc ${PROCESS_NUM}] [--coco-file ${COCO_PRED_FILE}]
```

Optional arguments:
- `RESULT_FILE`: Filename of the output results in pickle format.
- `TASK_NAME`: Task names in one of [`det`, `ins_seg`, `box_track`, `seg_track`]
- `BDD_OUPPUT_DIR`: The dir path to save the converted bdd jsons and masks.
- `COCO_PRED_FILE`: Filename of the json in coco submission format.
