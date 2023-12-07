import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR, YOLO

if __name__ == '__main__':
    # model = YOLO('/workspace/cv-docker/joey04.li/datasets/master_thesis_rtdetr/ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-Ortho.yaml')
    # model.load('')
    # model = RTDETR('/workspace/cv-docker/joey04.li/datasets/master_thesis_rtdetr/ultralytics/cfg/models/yolo-detr/yolov8-detr-fasternet.yaml')
    model = RTDETR('/workspace/cv-docker/joey04.li/datasets/master_thesis_rtdetr/ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-Ortho.yaml')
    # model.load('/workspace/cv-docker/joey04.li/datasets/master_thesis_rtdetr/weights/rtdetr_r18vd_5x_coco_objects365_from_paddle.pt') # loading pretrain weights
    model.train(data='/workspace/cv-docker/joey04.li/datasets/master_thesis_rtdetr/dataset/classroom_data.yaml',
                cache=False,
                imgsz=640,
                epochs=600,
                batch=1,
                workers=4,
                device='0',
                # resume='', # last.pt path
                project='/workspace/cv-docker/joey04.li/datasets/master_thesis_rtdetr/outputs/train',
                name='yolov8-detr-C2f-Ortho-60bs-600ep',
                exist_ok=True
                )