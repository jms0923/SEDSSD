import time
import argparse
import os
import sys
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import yaml
from PIL import Image
import cv2

from utils.anchor import generate_default_boxes
from dataset.box_utils import decode, compute_nms
from dataset.voc_data import create_batch_generator
from dataset.image_utils import ImageVisualizer
from utils.losses import create_losses
from utils.metric import iou

from networks.network import create_ssd
from networks.resnet101_network import create_dssd
from networks.senet_network import create_sedssd


parser = argparse.ArgumentParser()
# dataset args
parser.add_argument("--data-dir", default="/home/globus/minSeok/dataset/only_LG/")
parser.add_argument("--save-dir", default="outputs/images")
parser.add_argument("--data-year", default="2007")
# dssd args
parser.add_argument("--arch", default="sedssd1080")
parser.add_argument("--num-examples", default=-1, type=int)
parser.add_argument("--pretrained-type", default="specified")    # specified
parser.add_argument("--checkpoint-dir")
parser.add_argument("--checkpoint-path", default="/home/globus/minSeok/DSSD_tf2/trainer/checkpoints/sedssd1080/network400")
# efficient net args
parser.add_argument("--effi", default=True, type=bool)
parser.add_argument("--efficientnet-checkpoint-path", default="/home/globus/minSeok/DSSD_tf2/trainer/checkpoints/model_B1")
# cuda args
parser.add_argument("--gpu-id", default="0")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


def predict(imgs, default_boxes):
    confs, locs = network(imgs)

    confs = tf.squeeze(confs, 0)
    locs = tf.squeeze(locs, 0)

    confs = tf.math.softmax(confs, axis=-1)
    classes = tf.math.argmax(confs, axis=-1)
    scores = tf.math.reduce_max(confs, axis=-1)

    boxes = decode(default_boxes, locs)

    out_boxes = []
    out_labels = []
    out_scores = []

    for c in range(1, BASE_CONFIG['sedssd_num_classes']):
        cls_scores = confs[:, c]

        score_idx = cls_scores > BASE_CONFIG['threshold']
        cls_boxes = boxes[score_idx]
        cls_scores = cls_scores[score_idx]
        nms_idx = compute_nms(cls_boxes, cls_scores, 0.45, 200)
        cls_boxes = tf.gather(cls_boxes, nms_idx)
        cls_scores = tf.gather(cls_scores, nms_idx)
        cls_labels = [c] * cls_boxes.shape[0]

        out_boxes.append(cls_boxes)
        out_labels.extend(cls_labels)
        out_scores.append(cls_scores)

    out_boxes = tf.concat(out_boxes, axis=0)
    out_scores = tf.concat(out_scores, axis=0)

    boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
    classes = np.array(out_labels)
    scores = out_scores.numpy()

    return boxes, classes, scores


if __name__ == "__main__":
    with open("./config.yml") as f:
        cfg = yaml.load(f)
        
    try:
        config = cfg[args.arch.upper()]
        global BASE_CONFIG
        BASE_CONFIG = cfg['BASE']

    except AttributeError:
        raise ValueError("Unknown architecture: {}".format(args.arch))

    default_boxes = generate_default_boxes(config)

    batch_generator, info = create_batch_generator(
        root_dir=args.data_dir,
        year=args.data_year,
        default_boxes=default_boxes,
        new_size=config["image_size"],
        batch_size=BASE_CONFIG['batch_size'],
        num_batches=args.num_examples,
        labels=cfg['LABELS']['sedssd'],
        mode="test"
    )

    try:
        if 'se' in args.arch:
            network = create_sedssd(
                BASE_CONFIG['sedssd_num_classes'],
                args.arch,
                args.pretrained_type,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_path=args.checkpoint_path,
                config=config
            )
        elif 'dssd' in args.arch:
            network = create_dssd(
                BASE_CONFIG['sedssd_num_classes'],
                args.arch,
                args.pretrained_type,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_path=args.checkpoint_path,
                config=config
            )
        else:
            network = create_ssd(
                BASE_CONFIG['sedssd_num_classes'],
                args.arch,
                args.pretrained_type,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_path=args.checkpoint_path,
                config=config
            )
    except Exception as ValueError:
        print(ValueError)
        print("The program is exiting...")
        sys.exit()

    os.makedirs("outputs/images", exist_ok=True)
    os.makedirs("outputs/detects", exist_ok=True)

    if args.effi:
        effi = tf.keras.models.load_model(args.efficientnet_checkpoint_path)
        effiInfo = info["idx_to_name"] + cfg['LABELS']['efficientnet']
        visualizer = ImageVisualizer(effiInfo, save_dir=args.save_dir)
    else:
        visualizer = ImageVisualizer(info["idx_to_name"], save_dir=args.save_dir)

    for idx, (filename, imgs, gt_confs, gt_locs) in enumerate(tqdm(batch_generator, total=info["length"], desc="Testing...", unit="images")):
        start = time.time()
        filename = filename.numpy()[0].decode()

        boxes, classes, scores = predict(imgs, default_boxes)
        classes -= 1

        original_image = cv2.imread(os.path.join(info["image_dir"], "{}.jpg".format(filename)), cv2.IMREAD_COLOR)

        origin_h, origin_w, _ = original_image.shape
        boxes[:,:1] *= origin_w
        boxes[:,1:2] *= origin_h
        boxes[:,2:3] *= origin_w
        boxes[:,3:4] *= origin_h

        print('sedssd time : ', round((time.time() - start)*1000), 'ms')
        if args.effi:
            start = time.time()
            # mask
            noEffiMask = (classes == 3) + (classes == 4)
            yesEffiMask = np.logical_not(noEffiMask)

            noEffiMaskedClasses = classes[noEffiMask]
            noEffiMaskedScore = scores[noEffiMask]
            noEffiMaskedBoxes = boxes[noEffiMask, :]

            yesEffiMaskedScore = scores[yesEffiMask]
            yesEffiMaskedBoxes = boxes[yesEffiMask, :]

            if len(yesEffiMaskedBoxes) > 0:
                croppedBoxes, checkedOriginBoxes = visualizer.cropBoxes(original_image, yesEffiMaskedBoxes, filename, BASE_CONFIG['effi_input'])
                
                effiPred = effi.predict(croppedBoxes)
                crpBoxLabel = np.argmax(effiPred, axis=1)
                crpBoxLabel += 7

                saveBoxes = np.concatenate((noEffiMaskedBoxes, checkedOriginBoxes), axis=0)
                saveClasses = np.concatenate((noEffiMaskedClasses, crpBoxLabel), axis=0)
                saveScores = np.concatenate((noEffiMaskedScore, yesEffiMaskedScore), axis=0) # have to revise efficientnet score
            else:
                # no yesEffiMaskedBoxes
                pass
            print('effi time : ', round((time.time() - start)*1000), 'ms')

            visualizer.save_image_cv(original_image, saveBoxes, saveClasses, scores, "{}.jpg".format(filename))

        else:
            visualizer.save_image_cv(original_image, boxes, classes, scores, "{}.jpg".format(filename))

            # log result
            log_file = os.path.join("outputs/detects", "{}.txt")
            for cls, box, score in zip(classes, boxes, scores):
                cls_name = info["idx_to_name"][cls - 1]
                with open(log_file.format(cls_name), "a") as f:
                    f.write(
                        "{} {} {} {} {} {}\n".format(
                            filename, score, *[coord for coord in box]
                        )
                    )
