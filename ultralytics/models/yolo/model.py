# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.model import Model
from ultralytics.models import yolo  # noqa
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, PoseModel, SegmentationModel


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model='yolov8n.pt') -> None:
        """
        Initializes the YOLO model with the given pre-trained model file. Supports .pt and .yaml formats.

        Args:
            model (str): Path to the pre-trained model. Defaults to 'yolov8n.pt'.

        Raises:
            NotImplementedError: If the model file extension is not 'pt', 'yaml', or 'yml'.
        """
        if model and model.split('.')[-1] not in ('pt', 'yaml', 'yml'):
            raise NotImplementedError('YOLO only supports creating from *.pt, *.yaml, or *.yml files.')
        super().__init__(model=model, task='detect')  # 由于修改了yolov8的head 不指定task会导致guess_model_task方法无法智能识别任务类型

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            'classify': {
                'model': ClassificationModel,
                'trainer': yolo.classify.ClassificationTrainer,
                'validator': yolo.classify.ClassificationValidator,
                'predictor': yolo.classify.ClassificationPredictor, },
            'detect': {
                'model': DetectionModel,
                'trainer': yolo.detect.DetectionTrainer,
                'validator': yolo.detect.DetectionValidator,
                'predictor': yolo.detect.DetectionPredictor, },
            'segment': {
                'model': SegmentationModel,
                'trainer': yolo.segment.SegmentationTrainer,
                'validator': yolo.segment.SegmentationValidator,
                'predictor': yolo.segment.SegmentationPredictor, },
            'pose': {
                'model': PoseModel,
                'trainer': yolo.pose.PoseTrainer,
                'validator': yolo.pose.PoseValidator,
                'predictor': yolo.pose.PosePredictor, }, }
