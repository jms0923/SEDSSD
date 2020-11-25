import argparse
import os
import sys
import time
import yaml

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from dataset.voc_data import create_batch_generator
from utils.anchor import generate_default_boxes
from utils.losses import create_losses

from networks.network import create_ssd
from networks.resnet101_network import create_dssd
from networks.senet_network import create_sedssd


parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="/home/globus/minSeok/dataset/LSK_VOC")
parser.add_argument("--data-year", default="2007")
parser.add_argument("--arch", default="sedssd1080")
parser.add_argument("--batch-size", default=1, type=int)
parser.add_argument("--num-batches", default=-1, type=int)
parser.add_argument("--neg-ratio", default=3, type=int)
parser.add_argument("--initial-lr", default=1e-3, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight-decay", default=5e-4, type=float)
parser.add_argument("--num-epochs", default=8000, type=int)
parser.add_argument("--checkpoint-dir", default="/home/globus/minSeok/DSSD_tf2/trainer/checkpoints/sedssd1080_cluster3")
parser.add_argument("--checkpoint-path")
parser.add_argument("--pretrained-type", default="base")    # latest / specified / base
parser.add_argument("--gpu-id", default="0")
# parser.add_argument("--apex", default=True)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# if args.apex:
#     policy = mixed_precision.Policy('mixed_float16')
#     mixed_precision.set_policy(policy)
#     print('Compute dtype: %s' % policy.compute_dtype)
#     print('Variable dtype: %s' % policy.variable_dtype)


@tf.function
def train_step(imgs, gt_confs, gt_locs, network, criterion, optimizer):
    with tf.GradientTape() as tape:
        confs, locs = network(imgs)
        conf_loss, loc_loss = criterion(confs, locs, gt_confs, gt_locs)
        loss = conf_loss + loc_loss

    gradients = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, network.trainable_variables))

    return loss, conf_loss, loc_loss


if __name__ == "__main__":
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    with open("./config.yml") as f:
        cfg = yaml.load(f)

    try:
        config = cfg[args.arch.upper()]
        global BASE_CONFIG
        BASE_CONFIG = cfg['BASE']
        
    except AttributeError:
        raise ValueError("Unknown architecture: {}".format(args.arch))

    default_boxes = generate_default_boxes(config)

    print("args.data_dir : ", args.data_dir)

    batch_generator, val_generator, info = create_batch_generator(
        root_dir=args.data_dir,
        year=args.data_year,
        default_boxes=default_boxes,
        new_size=config["image_size"],
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        labels=cfg['LABELS']['sedssd'],
        mode="train",
        # augmentation=["flip"],
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
        
    except Exception as e:
        print(e)
        print("The program is exiting...")
        sys.exit()

    criterion = create_losses(args.neg_ratio, BASE_CONFIG['sedssd_num_classes'])

    steps_per_epoch = info["length"] // args.batch_size

    lr_fn = PiecewiseConstantDecay(
        boundaries=[
            int(steps_per_epoch * args.num_epochs * 2 / 3),
            int(steps_per_epoch * args.num_epochs * 5 / 6),
        ],
        values=[args.initial_lr, args.initial_lr * 0.1, args.initial_lr * 0.01],
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_fn, momentum=args.momentum)

    train_log_dir = "logs/train"
    val_log_dir = "logs/val"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    for epoch in range(args.num_epochs):
        avg_loss = 0.0
        avg_conf_loss = 0.0
        avg_loc_loss = 0.0
        start = time.time()
        for i, (_, imgs, gt_confs, gt_locs) in enumerate(batch_generator):
            loss, conf_loss, loc_loss = train_step(                         # , l2_loss
                imgs, gt_confs, gt_locs, network, criterion, optimizer
            )
            avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)
            avg_conf_loss = (avg_conf_loss * i + conf_loss.numpy()) / (i + 1)
            avg_loc_loss = (avg_loc_loss * i + loc_loss.numpy()) / (i + 1)
        print(
            "Epoch: {} Batch {} Time: {:.2}s | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f}".format(
                epoch + 1,
                i + 1,
                time.time() - start,
                avg_loss,
                avg_conf_loss,
                avg_loc_loss,
            )
        )

        avg_val_loss = 0.0
        avg_val_conf_loss = 0.0
        avg_val_loc_loss = 0.0

        for i, (_, imgs, gt_confs, gt_locs) in enumerate(val_generator):
            val_confs, val_locs = network(imgs)
            val_conf_loss, val_loc_loss = criterion(
                val_confs, val_locs, gt_confs, gt_locs
            )
            val_loss = val_conf_loss + val_loc_loss
            avg_val_loss = (avg_val_loss * i + val_loss.numpy()) / (i + 1)
            avg_val_conf_loss = (avg_val_conf_loss * i + val_conf_loss.numpy()) / (
                i + 1
            )
            avg_val_loc_loss = (avg_val_loc_loss * i + val_loc_loss.numpy()) / (i + 1)

        with train_summary_writer.as_default():
            tf.summary.scalar("loss", avg_loss, step=epoch)
            tf.summary.scalar("conf_loss", avg_conf_loss, step=epoch)
            tf.summary.scalar("loc_loss", avg_loc_loss, step=epoch)

        with val_summary_writer.as_default():
            tf.summary.scalar("loss", avg_val_loss, step=epoch)
            tf.summary.scalar("conf_loss", avg_val_conf_loss, step=epoch)
            tf.summary.scalar("loc_loss", avg_val_loc_loss, step=epoch)

        if (epoch + 1) % 50 == 0:
            network.save_weights(os.path.join(args.checkpoint_dir, "network{}".format(epoch + 1)))

