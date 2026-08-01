import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ensemble_boxes import *
import optuna
from collections import defaultdict
import torch
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances

"""
    Notes: 

    The default settings in YOLO validation are:
    1. IoU Threshold: 0.7
    2. Conf Threshold (skip box threshold): 1e-3

    Adhere to the same settings in weighted boxes fusion to remain consistent in the experimentation.

"""
def plots(study):
    plots_folder = "./plots"

    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder, exist_ok=True)

    plot_optimization_history(study)
    plt.savefig(os.path.join(plots_folder, "opt_history.png"), bbox_inches="tight")
    plt.close() 

    plot_intermediate_values(study)
    plt.savefig(os.path.join(plots_folder, "opt_intermediate_values.png"), bbox_inches="tight")
    plt.close()
    
    plot_parallel_coordinate(study)
    plt.savefig(os.path.join(plots_folder, "opt_parallel_coordinates.png"), bbox_inches="tight")
    plt.close()

    plot_param_importances(study)
    plt.savefig(os.path.join(plots_folder, "opt_param_importances.png"), bbox_inches="tight")
    plt.close()

    plot_contour(study)
    plt.savefig(os.path.join(plots_folder, "opt_contours.png"), bbox_inches="tight")
    plt.close()

def visualize(idx, image_path, gt_label, boxes):
    output_folder = "./predictions"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    image = cv2.imread(image_path)

    image = cv2.rectangle(image, (int(gt_label[0]), int(gt_label[1])), (int(gt_label[2]), int(gt_label[3])), color=(0, 0, 255), thickness=3)

    for box in boxes:
        image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(255, 0, 0), thickness=3)

    cv2.imwrite(os.path.join(output_folder, f"output_{idx}.png"), image)

def read_gt(images_path: str, labels_path: str):

    images_paths_list = sorted([os.path.join(images_path, x) for x in os.listdir(images_path)])
    labels_paths_list = sorted([os.path.join(labels_path, x) for x in os.listdir(labels_path)])
    
    ################# Debugging ###################
    # print(f"Entry with empty prediction: {labels_paths_list[230]}")
    ###############################################
    
    gt_labels  = []

    # List to store the class label which is the first entry in the yolo label.
    class_labels = []

    # Loop through the YOLO txt files
    for i, filepath in enumerate(labels_paths_list):
        image = cv2.imread(images_paths_list[i])

        image_height, image_width, _ = image.shape 
        
        with open(filepath, "r") as file:
            """
                YOLO label structure:

                class_label x_center y_center width_of_box height_of_box
            """            
            content = file.readlines()[0].strip() # Remove any whitespace or newline character using strip()

            class_labels.append(int(content.split(" ")[0])) # Collect the Class label

            # Rescale the normalized values to absolute.
            abs_x_center = float(content.split(" ")[1]) * image_width
            abs_y_center = float(content.split(" ")[2]) * image_height
            width = float(content.split(" ")[3]) * image_width
            height = float(content.split(" ")[4]) * image_height

            # Calculate (x_min, y_min, x_max, y_max) of the box.
            x_min = abs_x_center - (width / 2)
            x_max = abs_x_center + (width / 2)
            y_min = abs_y_center - (height / 2)
            y_max = abs_y_center + (height / 2)

            gt = [x_min, y_min, x_max, y_max]


        gt_labels.append(gt) # Contains the labels of all the images.

    
    return class_labels, gt_labels


def yolo_predictions(weights_path: list, images_path: str, opt):
    """ 
        Helper function when the system predicts only using YOLO models

        weights_path: path to the directory with all the YOLO models.
        images_path: Path to the directory with all the test images.
    """
    
    image_paths_list = sorted([os.path.join(images_path, x) for x in os.listdir(images_path)])

    # List to store boxes from all the images
    agg_boxes_list = []
    agg_scores_list = []
    agg_labels_list = []
    
    models = []
    for j, model_path in enumerate(weights_path): 
        models.append(YOLO(model_path))

    for i, path in enumerate(image_paths_list):
        # Collect the info from all the models
        boxes_list = []
        scores_list = []
        labels_list = []

        for model in models:
            results = model.predict(path, verbose=False)
            
            # Each result object corresponds to predictions from one image.
            for result in results:
                boxes = result.boxes.xyxyn.cpu().detach().numpy() # Returns all the bounding box corresponding to that image.
                scores = result.boxes.conf.cpu().detach().numpy()
                labels = result.boxes.cls.cpu().detach().numpy()

            boxes_list.append(boxes.tolist())
            scores_list.append(scores.tolist())
            labels_list.append(labels.tolist())
        
        agg_boxes_list.append(boxes_list)
        agg_scores_list.append(scores_list)
        agg_labels_list.append(labels_list)

    return agg_boxes_list, agg_scores_list, agg_labels_list

def ensemble(boxes_list, scores_list, labels_list, labels_path, images_path, opt):
    
    """
        Function to perform the weighted boxes fusion to the predictions.
        Here we treat the arguments weights and iou_thr as bayesian optimization problem.
    """
    # Images path just for viz purpose
    image_file_paths = sorted([os.path.join(images_path, x) for x in os.listdir(images_path)])

    ensembled_boxes = []
    ensembled_scores = []
    ensembled_labels = []

    len_models = 2

    # weights = [trial.suggest_float(f"weight_{i+1}", low=1.0, high=5.0, step=0.1) for i in range(len_models)]
    weights = [1.7, 3.7]
    # iou_thr = trial.suggest_float("iou_thr", low=0.5, high=1.0, step=0.1)

    # metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) 
    
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    ################ Remove Later ###################
    # weights = [1.5, 4.0] 
    iou_thr = 0.5
    #################################################

    class_labels, gt_labels = read_gt(images_path, labels_path)

    for i in range(len(boxes_list)):

        image = cv2.imread(image_file_paths[i])
        
        image_height, image_width, _ = image.shape
        
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list[i], scores_list[i], labels_list[i], weights=weights, iou_thr=iou_thr, skip_box_thr=opt.skip_box_thr
        )

        ensembled_boxes.append(boxes)
        ensembled_scores.append(scores)
        ensembled_labels.append(labels)

        # print(f"Shape of ensembled boxes: {np.array(boxes).shape}")
        if len(boxes) == 0:
            print(f"Empty Prediction in index - {i}")
            pred = [dict(
                boxes=torch.empty((0, 4), dtype=torch.float32),
                scores=torch.empty((0,), dtype=torch.float32),
                labels=torch.empty((0,), dtype=torch.int32)
            )]

        else:
            
            for box in boxes:
                box[0] = box[0] * image_width
                box[1] = box[1] * image_height
                box[2] = box[2] * image_width
                box[3] = box[3] * image_height

            # print(f"Original Boxes: {boxes_list[i]}")
            # print(f"Ground Truth: {gt_labels[i]}")
            # print(f"Ensembled Box: {boxes}")

            pred = [dict(
                boxes=torch.tensor(boxes, dtype=torch.float32),
                scores=torch.tensor(scores, dtype=torch.float32),
                labels=torch.tensor(labels, dtype=torch.int32)
            )]

            if opt.show:
                visualize(i, image_file_paths[i], gt_labels[i], boxes)

        target = [dict(
            boxes=torch.tensor([gt_labels[i]], dtype=torch.float32),
            labels=torch.tensor([class_labels[i]], dtype=torch.int32)
        )]

        metric.update(pred, target)
    
    final_metric = metric.compute()

    # trial.set_user_attr("Averaged mAP (50-95)", final_metric["map"])

    return final_metric["map_50"], final_metric["map"]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--images_path", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)

    ## WBF settings
    parser.add_argument("--skip_box_thr", type=float, default=1e-3)

    ## Bayesian Opt Settings
    parser.add_argument("--trials", type=int, default=100)

    ## Visualization settings
    parser.add_argument("--show", type=bool, default=False, help="Argument to visualize and save the images with the ground truth labels and the labels after applying WBF.")
    parser.add_argument("--plots", type=bool, default=True, help="Saves the plots related to the optimization.")

    opt = parser.parse_args()


    print(f"Cuda enabled Torch: {torch.cuda.is_available()}")

    weight_file_paths = sorted([os.path.join(opt.weights_path, x) for x in os.listdir(opt.weights_path)])

    all_boxes, all_scores, all_labels = yolo_predictions(weight_file_paths, opt.images_path, opt)

    map_50, map_50_95 = ensemble(all_boxes, all_scores, all_labels, opt.labels_path, opt.images_path, opt)

    # maps = ensemble(all_boxes, all_scores, all_labels, opt.labels_path, opt.images_path, opt)

    # print(f"\nmAp: {maps}")

    # study = optuna.create_study(direction="maximize")

    # study.optimize(
    #     lambda trial: ensemble(trial, all_boxes, all_scores, all_labels, opt.labels_path, opt.images_path, opt),
    #     n_trials=opt.trials
    # )

    # print(f"Best Value (mAP50): {study.best_trial.value}")
    # print(f"Best params: {study.best_trial.params}")

    # print(f"Corresponding mAP(50-95): {study.best_trial.user_attrs["Averaged mAP (50-95)"]}")

    print(f"mAP50: {map_50}")
    print(f"mAP50-95: {map_50_95}")

if __name__ == "__main__":
    main()