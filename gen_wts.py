import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Normalize
from models import hmr, SMPL
from utils.imutils import crop
import config
import constants
import struct




    
# class HMR:
#     def __init__(self, weight_path):
#         self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#         self.model = hmr(config.SMPL_MEAN_PARAMS).to(self.device)
#         checkpoint = torch.load(weight_path)
#         self.model.load_state_dict(checkpoint['model'], strict=False)
#         self.smpl = SMPL(config.SMPL_MODEL_DIR,
#                 batch_size=1,
#                 create_transl=False).to(self.device)
#         self.model.eval()

#     def process_image(self, img,  input_res=224):
#         normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         height = img.shape[0]
#         width = img.shape[1]
#         center = np.array([width // 2, height // 2])
#         scale = max(height, width) / 200
        
#         img = crop(img, center, scale, (input_res, input_res))
#         img = img.astype(np.float32) / 255.
#         img = torch.from_numpy(img).permute(2,0,1)
#         norm_img = normalize_img(img.clone())[None]
#         return img, norm_img
    
#     def run(self, img):
#         img, norm_img = self.process_image(img, input_res=constants.IMG_RES)
#         with torch.no_grad():
#             pred_rotmat, pred_betas, pred_camera = self.model(norm_img.to(self.device))
#             # pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
#             # pred_vertices = pred_output.vertices.squeeze(0).cpu().numpy()
#             # pred_faces = self.smpl.faces.astype(np.int32)

#         pred_rotmat = pred_rotmat.cpu().numpy()
#         pred_batch_euler_angles = []
#         for person in pred_rotmat:
#             euler_angles = []
#             for rot in person:
#                 euler_angle = rotationMatrixToEulerAngles(rot)
#                 euler_angles.append(euler_angle)
#             pred_batch_euler_angles.append(euler_angles)

#         return pred_batch_euler_angles

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load('data/model_checkpoint.pt')
    model.load_state_dict(checkpoint['model'], strict=False)
    smpl = SMPL(config.SMPL_MODEL_DIR,
            batch_size=1,
            create_transl=False).to(device)
    model.eval()

    with torch.no_grad():
        tmp = torch.ones(1, 3, 224, 224).to('cuda:0')
        pred_rotmat, pred_betas, pred_camera = model(tmp)
        print(pred_camera)
    
    #
    f = open("data/hmr.wts", 'w')
    f.write("{}\n".format(len(model.state_dict().keys())))
    for k,v in model.state_dict().items():
        # print('key: ', k)
        # print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
        

if __name__ == '__main__':
    main()