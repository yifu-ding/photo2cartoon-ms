import os
import cv2
# import torch
import numpy as np
from models import ResnetGenerator
import argparse
from utils import Preprocess
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor

from mindspore import Tensor, export, load_checkpoint, load_param_into_net
import mindspore as ms

parser = argparse.ArgumentParser()
parser.add_argument('--photo_path', type=str, help='input photo path')
parser.add_argument('--save_path', type=str, help='cartoon save path')
args = parser.parse_args()

os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

class Photo2Cartoon:
    def __init__(self):
        self.pre = Preprocess()

        self.net = ResnetGenerator(ngf=32, img_size=256, light=True)
        
        assert os.path.exists('./pretrained_models/photo2cartoon_weights_genA2B.ckpt'), "[Step1: load weights] Can not find 'photo2cartoon_weights.pt' in folder 'models!!!'"
        # params = pytorch2mindspore(params)
        # import pdb; pdb.set_trace()
        # param_dict = torch.load("./pretrained_models/photo2cartoon_weights.pt")
        # self.net.load_state_dict(param_dict['genA2B'])

        param_dict = load_checkpoint('./pretrained_models/photo2cartoon_weights_genA2B.ckpt')
        load_param_into_net(self.net, param_dict)
        print('[Step1: load weights] success!')

    def inference(self, img):
        # face alignment and segmentation
        face_rgba = self.pre.process(img)
        if face_rgba is None:
            print('[Step2: face detect] can not detect face!!!')
            return None
        
        print('[Step2: face detect] success!')
        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
        face = face_rgba[:, :, :3].copy()
        mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
        face = (face*mask + (1-mask)*255) / 127.5 - 1

        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
        face = Tensor(face)#.to(self.device)

        # inference
        # with torch.no_grad():
        # with ms.ops.stop_gradient(face):
        cartoon = self.net(face)[0][0]
        # import pdb; pdb.set_trace()
        # post-process
        cartoon = np.transpose(cartoon.asnumpy(), (1, 2, 0))
        cartoon = (cartoon + 1) * 127.5
        cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        print('[Step3: photo to cartoon] success!')
        return cartoon


# class Photo2Cartoon:
#     def __init__(self):
#         self.pre = Preprocess()

#         self.net = ResnetGenerator_TORCH(ngf=32, img_size=256, light=True)
        
#         assert os.path.exists('./pretrained_models/photo2cartoon_weights_genA2B.ckpt'), "[Step1: load weights] Can not find 'photo2cartoon_weights.pt' in folder 'models!!!'"
#         # params = pytorch2mindspore(params)
#         # import pdb; pdb.set_trace()
#         param_dict = torch.load("./pretrained_models/photo2cartoon_weights.pt")
#         self.net.load_state_dict(param_dict['genA2B'])

#         # param_dict = load_checkpoint('./pretrained_models/photo2cartoon_weights_genA2B.ckpt')
#         # load_param_into_net(self.net, param_dict)
#         print('[Step1: load weights] success!')

#     def inference(self, img):
#         # face alignment and segmentation
#         face_rgba = self.pre.process(img)
#         if face_rgba is None:
#             print('[Step2: face detect] can not detect face!!!')
#             return None
        
#         print('[Step2: face detect] success!')
#         face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
#         face = face_rgba[:, :, :3].copy()
#         mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
#         face = (face*mask + (1-mask)*255) / 127.5 - 1

#         face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
#         face = torch.Tensor(face)#.to(self.device)

#         # inference
#         # with torch.no_grad():
#         # with ms.ops.stop_gradient(face):
#         cartoon = self.net(face)[0][0]
#         import pdb; pdb.set_trace()
#         # post-process
#         cartoon = np.transpose(cartoon.detach().numpy(), (1, 2, 0))
#         cartoon = (cartoon + 1) * 127.5
#         cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
#         cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
#         print('[Step3: photo to cartoon] success!')
#         return cartoon


if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread(args.photo_path), cv2.COLOR_BGR2RGB)
    c2p = Photo2Cartoon()
    cartoon = c2p.inference(img)
    if cartoon is not None:
        cv2.imwrite(args.save_path, cartoon)
        print('Cartoon portrait has been saved successfully!')
