import numpy as np
from sklearn import metrics


def iou(true_boxes, pred_boxes):
    scores = metrics.jaccard_score(true_boxes, pred_boxes)
    print(scores)