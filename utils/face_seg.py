import os
import cv2
import numpy as np
# import tensorflow as tf
# from tensorflow.python.platform import gfile
import torch.nn as nn
import torch
# from models.ms.fcn8s import FCN8s
# from models.fcn8s_torch import FCN8s as FCN8s_TORCH
# curPath = os.path.abspath(os.path.dirname(__file__))
# import mindspore_hub 
# from utils.utils import pytorch2mindspore_fcn
from mindspore import Tensor, export, load_checkpoint, load_param_into_net


class FaceSeg:
    def __init__(self, model_path=os.path.join('save_model', 'FCN.pth')):
        # self.seg = FCN(num_classes=2, backbone=HRNet_W18())
        # torch_state_dict = torch.load(model_path)
        # self.seg.load_state_dict(torch_state_dict, strict=True)

        # import torch
        self.seg = torch.hub.load('pytorch/vision:v0.8.2', 'fcn_resnet101', pretrained=True)
        self.seg.eval()

        # model_uid = "mindspore/ascend/0.7/googlenet_v1_cifar10"  # using GoogleNet as an example.
        # model_uid = "mindspore/1.7/facedetection_widerface"
        # model_ckpt = "/home/yifu/personal/study/ai_course/pretrained_models/fcn8s_ascend_v170_voc2012_official_cv_meanIoU64.57.ckpt"
        # network = mindspore_hub.load(name=model_uid,  pretrained=False)
        # model_ckpt = load_checkpoint(model_ckpt)
        # load_param_into_net(network, model_ckpt)        
        # # net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        # net_loss = loss.SoftmaxCrossEntropyLoss(cfg.num_classes, cfg.ignore_label)
        # self.seg = Model(network, net_loss)

        # model_pth = "./pretrained_models/fcn8s-heavy-pascal.pth"
        # torch_model = torch.load(model_pth)
        # self.seg = FCN8s_TORCH()
        # self.seg.load_state_dict(torch_model)
        # self.seg.eval()
        # pytorch2mindspore_fcn(torch_model)
        # import pdb; pdb.set_trace()

        # model_pth = "./pretrained_models/fcn8s.ckpt"
        # model_ckpt = load_checkpoint(model_pth)
        # self.seg = FCN8s()
        # load_param_into_net(self.seg, model_ckpt)      

        
    def input_transform(self, image):
        image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
        image = (image / 255.)[np.newaxis, :, :, :]
        image = np.transpose(image, (0, 3, 1, 2)).astype(np.float32)
        image_input = torch.Tensor(image)
        return image_input

    def output_transform(self, output, shape):
        output = cv2.resize(output, (shape[1], shape[0]))
        image_output = (output * 255).astype(np.uint8)
        return image_output

    def get_mask(self, image):
        image_input = self.input_transform(image)
        with torch.no_grad():
            logits = self.seg(image_input)['out'][0]
            # logits = self.seg(image_input)[0]
        # pred = torch.argmax(logits[0], axis=1)
        # import pdb; pdb.set_trace()
        pred = logits.argmax(0)
        pred = pred.numpy()
        mask = np.squeeze(pred).astype('uint8')

        mask = self.output_transform(mask, shape=image.shape[:2])
        return mask


# class FaceSeg:
#     def __init__(self, model_path=os.path.join("pretrained_models", 'seg_model_384.pb')):
#         config = tf.compat.v1.ConfigProto()
#         config.gpu_options.allow_growth = True
#         self._graph = tf.Graph()
#         self._sess = tf.compat.v1.Session(config=config, graph=self._graph)

#         self.pb_file_path = model_path
#         self._restore_from_pb()
#         self.input_op = self._sess.graph.get_tensor_by_name('input_1:0')
#         self.output_op = self._sess.graph.get_tensor_by_name('sigmoid/Sigmoid:0')

#     def _restore_from_pb(self):
#         with self._sess.as_default():
#             with self._graph.as_default():
#                 with gfile.FastGFile(self.pb_file_path, 'rb') as f:
#                     graph_def = tf.compat.v1.GraphDef()
#                     graph_def.ParseFromString(f.read())
#                     tf.import_graph_def(graph_def, name='')

#     def input_transform(self, image):
#         image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
#         image_input = (image / 255.)[np.newaxis, :, :, :]
#         return image_input

#     def output_transform(self, output, shape):
#         output = cv2.resize(output, (shape[1], shape[0]))
#         image_output = (output * 255).astype(np.uint8)
#         return image_output

#     def get_mask(self, image):
#         image_input = self.input_transform(image)
#         output = self._sess.run(self.output_op, feed_dict={self.input_op: image_input})[0]
#         return self.output_transform(output, shape=image.shape[:2])
