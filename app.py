import os
#os.system("wget https://github.com/hustvl/YOLOP/raw/main/weights/End-to-end.pth")
#os.system("wget https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt")
from pathlib import Path
import cv2
import torch

from utils.functions import \
        time_synchronized,select_device, increment_path,\
        scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
        driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,\
        LoadImages
      
import numpy as np
IMAGE_W = 1280
IMAGE_H = 233
src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
dst = np.float32([[3*IMAGE_W//7, IMAGE_H], [4*IMAGE_W//7, IMAGE_H], [0, 0], [IMAGE_W, 0]])
M = cv2.getPerspectiveTransform(src, dst) 
Minv = cv2.getPerspectiveTransform(dst, src)
def birdeye(img):
    img_scaled = cv2.resize(img,(1280,720))
    global IMAGE_H,IMAGE_W,M,Minv
    img_croped = img_scaled[450:, 0:IMAGE_W]
    warped_img = cv2.warpPerspective(img_croped, M, (IMAGE_W, IMAGE_H))
    #result_img = cv2.line(warped_img,(640,0),(640,720),(0,0,0),5) 
    #return np.concatenate((result_img,color_selection(warped_img)),axis=1)
    return warped_img
def detect():
    conf_thres = 0.25
    iou_thres = 0.45
    weights = "yolopv2.pt"
    imgsz = 640
    save_img = True
    device = select_device('')
    half = device.type != 'cpu'
    stride = 32
    model  = torch.jit.load(weights,map_location=device)
    model.eval()
    #out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20, (1280,720))
    dataset = LoadImages("Inference/test.mp4", img_size=imgsz, stride=stride)
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float() 
        img /= 255.0 
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        [pred,anchor_grid],seg,ll= model(img)
        pred = split_for_trace_model(pred,anchor_grid)
        pred = non_max_suppression(pred,conf_thres,iou_thres, classes=None, agnostic=True)
        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)
        for i, det in enumerate(pred):
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)
            s += '%gx%g ' % img.shape[2:]  
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] 
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if save_img :
                        plot_one_box(xyxy, im0, line_thickness=3)
            show_seg_result(im0, (da_seg_mask,ll_seg_mask),is_demo=True)
        result = cv2.cvtColor(im0[:,:,::-1],cv2.COLOR_BGR2RGB)
        result = cv2.line(result,(len(result[0])//2,0),(len(result[0])//2,len(result)),(0,0,0),5)
        #out.write(result)
        cv2.imshow('result',birdeye(result))
        cv2.imshow('result0',result)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        print()
    print("video released")
    #out.release()
detect()
