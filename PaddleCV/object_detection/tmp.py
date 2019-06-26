import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.3"
import time
import numpy as np
import argparse
import functools
import math

import paddle
import paddle.fluid as fluid
import reader
from mobilenet_ssd import build_mobilenet_ssd
from utility import add_arguments, print_arguments
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 保存预测模型路径
save_path = 'model_0618/best_model/'
# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)

image_shape = [3, 300, 300]
num_classes = 81
batch_size = 8
data_dir = 'data/coco'
test_list = 'annotations/instances_val2017.json'
label_file = 'label_list'

data_args = reader.Settings(
        dataset='coco2017',
        data_dir=data_dir,
        label_file=label_file,
        resize_h=300,
        resize_w=300,
        mean_value=[127.5, 127.5, 127.5],
        apply_distort=False,
        apply_expand=False,
        ap_version='cocoMAP')

image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
gt_box = fluid.layers.data(
    name='gt_box', shape=[4], dtype='float32', lod_level=1)
gt_label = fluid.layers.data(
    name='gt_label', shape=[1], dtype='int32', lod_level=1)
gt_iscrowd = fluid.layers.data(
    name='gt_iscrowd', shape=[1], dtype='int32', lod_level=1)
gt_image_info = fluid.layers.data(
    name='gt_image_id', shape=[3], dtype='int32')

test_reader = reader.test(data_args, test_list, batch_size)
feeder = fluid.DataFeeder(
    place=place,
    feed_list=[image, gt_box, gt_label, gt_iscrowd, gt_image_info])

executor = fluid.Executor(place)
test_program = fluid.Program()
with fluid.program_guard(test_program):
    boxes = fluid.layers.data(
        name='boxes', shape=[-1,-1,4], dtype='float32')
    scores = fluid.layers.data(
        name='scores', shape=[-1,-1,num_classes], dtype='float32')
    pred_result = fluid.layers.multiclass_nms(
        bboxes=boxes,
        scores=scores,
        score_threshold=0.01,
        nms_top_k=-1,
        nms_threshold=0.45,
        keep_top_k=-1,
        normalized=False)

executor.run(fluid.default_startup_program())

cocoGt = COCO(os.path.join(data_args.data_dir, test_list))
json_category_id_to_contiguous_id = {
    v: i + 1
    for i, v in enumerate(cocoGt.getCatIds())
}
contiguous_category_id_to_json_id = {
    v: k
    for k, v in json_category_id_to_contiguous_id.items()
}

def get_dt_res(nmsed_out_v, data):
    dts_res = []
    lod = nmsed_out_v[0].lod()[0]
    nmsed_out_v = np.array(nmsed_out_v[0])
    real_batch_size = min(batch_size, len(data))
    assert (len(lod) == real_batch_size + 1), \
    "Error Lod Tensor offset dimension. Lod({}) vs. batch_size({})".format(len(lod), batch_size)
    k = 0
    for i in range(real_batch_size):
        dt_num_this_img = lod[i + 1] - lod[i]
        image_id = int(data[i][4][0])
        image_width = int(data[i][4][1])
        image_height = int(data[i][4][2])
        for j in range(dt_num_this_img):
            dt = nmsed_out_v[k]
            k = k + 1
            category_id, score, xmin, ymin, xmax, ymax = dt.tolist()
            xmin = max(min(xmin, 1.0), 0.0) * image_width
            ymin = max(min(ymin, 1.0), 0.0) * image_height
            xmax = max(min(xmax, 1.0), 0.0) * image_width
            ymax = max(min(ymax, 1.0), 0.0) * image_height
            w = xmax - xmin
            h = ymax - ymin
            bbox = [xmin, ymin, w, h]
            dt_res = {
                'image_id': image_id,
                'category_id': contiguous_category_id_to_json_id[category_id],
                'bbox': bbox,
                'score': score
            }
            dts_res.append(dt_res)
    return dts_res


dts_res = []

for batch_id, data in enumerate(test_reader()):
    boxes_np, socres_np = exe.run(program=infer_program,
                                  feed={feeded_var_names[0]: feeder.feed(data)['image']},
                                  fetch_list=target_var)

    nms_out = executor.run(
        program=test_program,
        feed={
            'boxes': boxes_np,
            'scores': socres_np
        },
        fetch_list=[pred_result], return_numpy=False)
    # nmsed_out_v = np.array(nms_out)
    if batch_id % 20 == 0:
        print("Batch {0}".format(batch_id))
    dts_res += get_dt_res(nms_out, data)


import tempfile
_, tmp_file = tempfile.mkstemp()
with open(tmp_file, 'w') as outfile:
    json.dump(dts_res, outfile)
print("start evaluate using coco api")
cocoGt = COCO(os.path.join(data_args.data_dir, test_list))
cocoDt = cocoGt.loadRes(tmp_file)
cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()