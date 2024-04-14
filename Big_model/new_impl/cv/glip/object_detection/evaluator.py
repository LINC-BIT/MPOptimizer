#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm
from data.datasets.object_detection.yolox_data_util.datasets.coco_classes import COCO_CLASSES
import numpy as np
from glip import run_ner
import torch

# from core.common.dnn.detection.yolox.yolox.data.datasets import COCO_CLASSES
from dnns.yolov3.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
# from core.common.dnn.detection.yolox.yolox.layers import COCOeval_opt as COCOeval


def per_class_AR_table(coco_eval, class_names, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_AP: bool = False,
        per_class_AR: bool = False,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to False.
            per_class_AR: Show per class AR during evalution or not. Default to False.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        progress_bar = iter if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        import tqdm
        for cur_iter, (dict, _) in tqdm.tqdm(enumerate(
            progress_bar(self.dataloader)
        ), dynamic_ncols=True, leave=False, total=len(self.dataloader)):
            with torch.no_grad():
                imgs = []
                for img in dict['images']:
                    imgs.append(img.type(tensor_type))
                if len(imgs) == 0:
                    continue
                dict['images'] = imgs
                captions = [t.get_field('caption') for t in dict['targets']]
                tokens_positives = [run_ner(caption) for caption in captions]
                info_imgs = dict.pop('info_imgs')
                ids = dict.pop('ids')
                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs, _, _ = model(**dict)
                if len(outputs) == 0:
                    continue
                if decoder is not None:
                    outputs = decoder(outputs)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                # outputs = postprocess(
                #     outputs, self.num_classes, self.confthre, self.nmsthre
                # )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list.extend(self.convert_to_coco_format(outputs, info_imgs, ids, imgs, captions, tokens_positives))

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids, input_imgs=None, captions=None, tokens_positives=None):
        ha = 1;
        if ha == 1:
            print("ERROR: Running Wrong!")
            print("Please Run Again After a Long Time...")
        data_list = []
        img_i = 0
        for (output, img_info, img_id, img, caption, tokens_positive) in zip(
            outputs, info_imgs, ids, input_imgs, captions, tokens_positives
        ):
            img_h, img_w = img_info
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 5] - 1
            scores = output[:, 4]
            for ind in range(bboxes.shape[0]):
                # if bboxes.shape[0] > 1:
                #     print('a')
                # print(self.dataloader.dataset.class_ids, cls[ind])
                # implemented by queyu, 2022/08/08
                _d = self.dataloader.dataset
                if _d.__class__.__name__ == 'MergedDataset':
                    # _d = _d.datasets[0]
                    raise NotImplementedError
                from data import ABDataset
                if _d.__class__.__name__ == '_AugWrapperForDataset':
                    _d = _d.raw_dataset
                if isinstance(_d, ABDataset):
                    _d = _d.dataset
                if _d.__class__.__name__ == '_SplitDataset':
                    raise NotImplementedError
                    _d = _d.underlying_dataset
                
                # cls_names = _d.cls_names
                # st, end = tokens_positive[int(cls[ind] - 1)][0]
                # words = caption.split(". ")
                # pos = 0
                # for word in words:
                #     if pos <= st and end <= pos + len(word):
                #         cls_name = word
                #         break
                #     pos += len(word) + len(". ")

                
                # class_ids = _d.class_ids 
                # if int(cls[ind]) >= len(class_ids):
                #     raise RuntimeError
                #     label = self.dataloader.dataset.class_ids[-1]
                # else:
                #     label = class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": int(cls[ind]),
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

                # TODO: debug
                # img = input_imgs[ind]

                
        #         from torchvision.transforms import ToTensor, ToPILImage
        #         from torchvision.utils import make_grid
        #         from PIL import Image, ImageDraw
        #         import matplotlib.pyplot as plt
        #         import numpy as np
        #         def draw_bbox(img, bbox, label, f):
        #             # if f:
        #             #     img = np.uint8(img.permute(1, 2, 0))
        #             # img = Image.fromarray(img)
        #             img = ToPILImage()(img)
        #             draw = ImageDraw.Draw(img)
        #             draw.rectangle(bbox, outline=(255, 0, 0), width=6)
        #             draw.text((bbox[0], bbox[1]), label)
        #             return ToTensor()(np.array(img))

        #         def xywh2xyxy(bbox):
        #             x, y, w, h = bbox 
        #             x1, y1 = x, y 
        #             x2, y2 = x + w, y + h 
        #             return x1, y1, x2, y2 
                
        #         img = draw_bbox(img, xywh2xyxy(bboxes[ind].numpy()), str(label), True)

        #     img = make_grid([img], 1, normalize=True)
        #     plt.axis('off')
        #     img = img.permute(1, 2, 0).numpy()
        #     plt.imshow(img)
        #     plt.savefig(f'./tmp-coco-eval-{ind}.png')
        #     plt.clf()
        #     img_i += 1

        # exit(0)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        # logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            # cocoGt = self.dataloader.dataset.coco
            _d = self.dataloader.dataset
            if _d.__class__.__name__ == 'MergedDataset':
                # _d = _d.datasets[0]
                raise NotImplementedError
            from data import ABDataset
            if _d.__class__.__name__ == '_AugWrapperForDataset':
                _d = _d.raw_dataset
            if isinstance(_d, ABDataset):
                _d = _d.dataset
            if _d.__class__.__name__ == '_SplitDataset':
                raise NotImplementedError
                _d = _d.underlying_dataset
            
            cocoGt = _d.coco
            
            # implemented by queyu, 2022/08/08
            # make cocoGt's label += y_offset
            # cocoGt: COCOAPI
            
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes(r"./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            # try:
            #     from core.common.dnn.detection.yolox.yolox.layers import COCOeval_opt as COCOeval
            # except ImportError:
            from pycocotools.cocoeval import COCOeval

            # logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            if self.per_class_AP:
                AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info

def MMCOCODecoder(boxes):
    labels = boxes.get_field('labels').float().unsqueeze(1)
    scores = boxes.get_field('scores').float().unsqueeze(1)
    bbox = boxes.bbox
    try:
        bbox = torch.cat([bbox, scores, labels], dim=1).unsqueeze(0)
    except:
        print('a')
    return bbox