# Taiji Pose Classification Project

This repository contains the implementation of a multimodal approach to Taiji human pose recognition, leveraging ResNet-18, MLPs for foot pressure data, and 3D MoCAP body joint coordinates. The PSUTMM100 dataset([link](http://vision.cse.psu.edu/data/data.shtml)), enriched with multiple modalities such as images, 3D MoCAP body joint coordinates, and foot pressure, was used for this project.

## Project Overview
The aim of the project is to classify a pose into one of the 45 key poses and 1 non-key pose frame in the PSUTMM100 dataset. The model's performance is evaluated using a Leave-One-Subject-Out (LOSO) approach, with the classification accuracy measured using precision, recall, and F1-score for all classes.

## Dataset Details
The PSUTMM100 dataset comprises 10 subjects, each with 10 takes, totaling 100 videos. The model is trained in a LOSO cross-validation manner, training on 9 subjects and testing on 1. The dataset contains 46 classes, with each pose represented by 100 data points from each subject. Notably, Subject3 Class 27 does not have any data points.

## Repository Contents
* `datasets_loaders.py`: Houses the TaijiDataset(Dataset) class used to create train and test datasets.
* `models.py`: Defines the models used - MLP, ResNet, and CombinedModel (enabling combinations of ResNet Only, MLP footp + MLP MoCap, and ResNet + MLP footp + MLP MoCap).
* `utils.py`: Includes helper functions for calculating metrics, plotting confusion matrix, and saving predictions.
* `main.py`: Trains 10 models in a LOSO fashion and saves the results in the corresponding folders.

## Instructions to Run
1. Ensure the PSUTMM100 dataset is available in your environment.
2. Update the `dataset_root_dir` variable in `main.py` with the path where your PSUTMM100 dataset resides.
3. Update the `results_root_dir` variable in `main.py` with the directory where you want the results saved.
4. Adjust the hyperparameters in `main.py` as needed, including `mocap_input_dim`, `footp_input_dim`, `num_epochs`, `learning_rate`, `num_classes`, and `batch_size`.
5. Run the `main.py` script to start training and evaluating the models. This script will train 10 models in a LOSO fashion and save results in the specified results directory.

```
python main.py
```

## Performance Summary
The following table summarizes the performance of different model variations.

| Model                          | Avg. Precision | Avg. Recall | Avg. f1-score |
|--------------------------------|----------------|-------------|---------------|
| ResNet Only                    | 0.87343        | 0.83232     | 0.82518       |
| MLP footp + MLP MoCap(baseline)| 0.39163        | 0.36146     | 0.32322       |
| ResNet + MLP footp + MLP MoCap | 0.87025        | 0.83381     | 0.83007       |

## Visualizations
Feature maps and T-SNE visualizations provide insightful information about the model's internal workings and effectiveness. Please refer to the following:

* Feature Maps of Block-1 of ResNet-18 from the ResNet + MLP footp + MLP MoCap model: ![Feature Maps](https://github.com/sujeethrv/prml_Taiji_Pose_classification/blob/main/VisualizationFiles/ResNet%20Feature%20Maps%20Layer%204_feature_maps.png)
* Train and Test T-SNE Visualizations of the outputs of the last fully connected layer for the ResNet + MLP footp + MLP MoCap model with Subject2 as test data: ![Train T-SNE Visualization](https://github.com/sujeethrv/prml_Taiji_Pose_classification/blob/main/VisualizationFiles/Subject2_train_tsne_visualization_train_test.png), ![Test T-SNE Visualization](https://github.com/sujeethrv/prml_Taiji_Pose_classification/blob/main/VisualizationFiles/Subject2_test_tsne_visualization_train_test.png)

Thank you for your interest in our Taiji Pose Classification Project.
