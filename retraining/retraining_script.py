import os
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.model_zoo import model_zoo

class CustomDatasetTrainer:
    def __init__(self, train_json, train_images, val_json, val_images):
        self.train_json = train_json
        self.train_images = train_images
        self.val_json = val_json
        self.val_images = val_images

    def register_datasets(self):
        register_coco_instances("custom_train", {}, self.train_json, self.train_images)
        register_coco_instances("custom_val", {}, self.val_json, self.val_images)

    def train_model(self):
        cfg = get_cfg()

        # Configuring Faster R-CNN with ResNet50 backbone
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Change to the number of classes in your dataset

        cfg.DATASETS.TRAIN = ("custom_train",)
        cfg.DATASETS.TEST = ("custom_val",)
        cfg.DATALOADER.NUM_WORKERS = 4

        # Set batch size and iterations for 1 epoch
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 1000  # Assuming 500 samples, so 1000 iterations for 1 epoch
        cfg.SOLVER.STEPS = (500,)

        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        # Optionally: Evaluate and use the trained model for further tasks
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        # Add code for evaluation or inference if needed

def main():
    train_json = "data/jsons/train.json"
    train_images = "data/images/train"
    val_json = "data/jsons/val.json"
    val_images = "data/images/val"

    custom_trainer = CustomDatasetTrainer(train_json, train_images, val_json, val_images)
    custom_trainer.register_datasets()
    custom_trainer.train_model()

if __name__ == "__main__":
    main()
