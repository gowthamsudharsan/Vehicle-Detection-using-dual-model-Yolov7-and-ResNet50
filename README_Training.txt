================================================================================
 Vehicle Detection and Scene Classification in Adverse Weather - TRAINING
================================================================================

This project trains two deep learning models for a comprehensive traffic analysis system:
1. A YOLOv7 model for object detection (vehicles, people, signs).
2. A ResNet50 model for scene classification (weather and lighting conditions).

The training is designed to be performed on the Kaggle platform to leverage their free GPU resources.

---------------------------------
I. TRAINING ENVIRONMENT
---------------------------------

- Platform: Kaggle Notebooks
- Accelerator: GPU (T4 or P100 recommended)
- Estimated Training Time: Approximately 6-8 hours on a P100 GPU.

NOTE: All necessary Python libraries (PyTorch, torchvision, etc.) are pre-installed in the Kaggle environment. No manual installation is required. The notebook itself will handle cloning the YOLOv7 repository and patching it for compatibility.

---------------------------------
II. DATASET
---------------------------------

This project uses a custom combined dataset sourced from BDD100K and DAWN.

1.  **BDD100K (10k Subset):**
    - **Download Link:** http://bdd-data.berkeley.edu/download.html
    - **Files to Download:** "100K Images" and "Labels" (Detection).
    - **Preparation:** A custom script (`filter.py` - provided separately) is used to randomly select 10,000 images and their corresponding 10,000 JSON labels from the full dataset.
    - **IMPORTANT:** Before running `filter.py`, you must open the script and edit the directory paths to match the location of your downloaded BDD100K data on your local machine.

2.  **DAWN (Dense Adverse Weather-condition Network):**
    - **Download Link:** https://data.mendeley.com/datasets/766ygrbt8y/3
    - **Preparation:** From the downloaded dataset, only the image files inside the `Fog`, `Rain`, `Snow`, and `Sand` folders are needed. Other folders like `Pascal VOC` or `YOLO_darknet` can be removed, as this dataset is only used for scene classification.

---------------------------------
III. STEPS TO RUN TRAINING ON KAGGLE
---------------------------------

1.  **Prepare the Dataset .zip File:**
    - On your local computer, create a root folder (e.g., "vehicle-dataset").
    - Inside it, create another folder named "Dataset".
    - Inside "Dataset", place your two prepared data folders: "BDD10K" (containing your 10k subset) and "DAWN".
    - The final structure should be:
      ```
      vehicle-dataset/
      └── Dataset/
          ├── BDD10K/
          │   ├── images10k/
          │   └── labels10k/
          └── DAWN/
              ├── Fog/
              ├── Rain/
              ...
      ```
    - Compress the top-level "vehicle-dataset" folder into a single .zip file (e.g., `vehicle-dataset.zip`).

2.  **Upload to Kaggle:**
    - Go to Kaggle (https://www.kaggle.com).
    - In the left menu, click on "Datasets", then click the "New Dataset" button.
    - Give your dataset a title (e.g., "Vehicle Detection Adverse Weather Data").
    - Drag and drop your `vehicle-dataset.zip` file into the upload area. Kaggle will automatically unzip it.
    - Wait for the upload and processing to finish.

3.  **Set up the Kaggle Notebook:**
    - Create a new Kaggle Notebook.
    - On the right-hand side panel, click **"+ Add data"**.
    - Find your newly uploaded dataset under the "Your Datasets" tab and add it to the notebook. It will be available at the path `/kaggle/input/vehicle-dataset/`.
    - In the right-hand panel, under "Notebook Options", set the **Accelerator** to **GPU**.

4.  **Run the Training:**
    - The entire training pipeline is contained within a single notebook file: `vehicle-detection-yolov7.ipynb`.
    - Copy the full content of the provided notebook into a single cell in your Kaggle environment.
    - Run the cell. The script will handle all data preparation, patching, model training, and saving.

---------------------------------
IV. EXPECTED OUTPUT
---------------------------------

After the ~6-8 hour run completes successfully, two high-quality model files will be generated in the notebook's output directory (`/kaggle/working/final_trained_models/`):

1.  **yolov7_object_detector.pt** (approx. 75 MB)
2.  **resnet50_scene_classifier.pth** (approx. 95 MB)

These two files are the final artifacts of the training process and are ready to be downloaded and used in the local deployment application (`Adverse_Weather_App`).

================================================================================