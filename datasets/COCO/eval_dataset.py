from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import argparse
import os

DATASET_PATH = "./val2017"
PREPROCESS_DATASET_PATH = "./val2017_preprocess/"
GT_JSON = "./annotations/instances_val2017.json"

def rescale(ori_shape, boxes, target_shape):
    """
        Rescale the output to the original image shape
    """
    ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
    padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

    boxes[:, [0, 2]] -= padding[0]
    boxes[:, [1, 3]] -= padding[1]
    boxes[:, :4] /= ratio

    boxes[:, 0] = min(boxes[:, 0], target_shape[1])  # x1
    boxes[:, 1] = min(boxes[:, 1], target_shape[0])  # y1
    boxes[:, 2] = min(boxes[:, 2], target_shape[1])  # x2
    boxes[:, 3] = min(boxes[:, 3], target_shape[0])  # y2

    return boxes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate COCO dataset")
    parser.add_argument("--result_json", type=str, help="Result JSON file path", required=True)
    args = parser.parse_args()

    result_json = args.result_json
    cocoGt = COCO(GT_JSON)
    results = json.load(open(result_json, 'r'))
    post_processed_results = []

    for item in results:
        image_id = item['image_id']
        shape_path = PREPROCESS_DATASET_PATH + str(image_id).zfill(12) + "_shape.npy"
        
        if not os.path.exists(shape_path):
            print(f"Shape file for image {image_id} does not exist. Skipping.")
            continue
        shape = np.load(shape_path)
        xywh = item['bbox']
        xyxy = np.array([[xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]], dtype=np.float32)
        xyxy = rescale([640, 640], xyxy, shape).round()[0]
        rescaled_bbox = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
        item['bbox'] = rescaled_bbox
        post_processed_results.append(item)
    cocoDt = cocoGt.loadRes(post_processed_results)

cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
