import warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from ultralytics import RTDETR, YOLO

if __name__ == '__main__':
    # model = YOLO('/workspace/cv-docker/joey04.li/datasets/master_thesis_rtdetr/ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-Ortho.yaml')
    # model.load('')
    # model = RTDETR('/workspace/cv-docker/joey04.li/datasets/master_thesis_rtdetr/ultralytics/cfg/models/yolo-detr/yolov8-detr-fasternet.yaml')
    model = RTDETR('/workspace/cv-docker/joey04.li/datasets/master_thesis_rtdetr/ultralytics/cfg/models/yolo-detr/yolov8s-detr-C2f-DCNV2-Dynamic.yaml')
    # model.load('/workspace/cv-docker/joey04.li/datasets/master_thesis_rtdetr/weights/rtdetr_r18vd_5x_coco_objects365_from_paddle.p/workspace/cv-docker/joey04.li/datasets/master_thesis_rtdetr/ultralytics/cfg/models/yolo-detr/yolov8-detr.yamlt') # loading pretrain weights
    model.train(data='/workspace/cv-docker/joey04.li/datasets/master_thesis_rtdetr/dataset/classroom_data.yaml',
                cache=False,
                imgsz=640,
                epochs=500,
                batch=128,
                workers=8,
                device='6',
                # resume='', # last.pt path
                project='/workspace/cv-docker/joey04.li/datasets/master_thesis_rtdetr/outputs/train',
                name='yolov8s-detr-C2f-DCNV2-Dynamic-128bs-500ep',
                exist_ok=True,
                patience=100,
                )