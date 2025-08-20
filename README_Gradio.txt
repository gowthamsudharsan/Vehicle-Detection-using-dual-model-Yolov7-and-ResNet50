==========================================================================
                    VEHICLE DETECTION IN ADVERSE WEATHER
                         Using YOLOv7 and ResNet50
==========================================================================

PROJECT OVERVIEW
================
This project is an AI-powered application that detects vehicles and classifies 
weather conditions from images and videos. It uses two deep learning models:

1. YOLOv7 - For object detection (vehicles, persons, traffic signs, etc.)
2. ResNet50 - For weather scene classification (Clear, Foggy, Rainy, etc.)

The application provides a user-friendly web interface built with Gradio that 
allows users to:
- Upload images for analysis
- Process videos frame by frame
- Use real-time webcam detection
- Identify potential hazards in adverse weather conditions

KEY FEATURES
============
✓ Multi-weather scene classification (Clear-Day, Foggy, Night, Overcast-Day, 
  Rainy, Sandstorm, Snowy)
✓ Real-time object detection with bounding boxes
✓ Anomaly detection for safety hazards (highlighted in red):
  - Persons on road
  - Traffic lights
  - Traffic signs
  - Bicycles
✓ Video processing with anti-flicker technology
✓ Live webcam detection
✓ Detection history logging
✓ Downloadable processed images and videos

HOW THE APPLICATION WORKS
==========================

app.py - Main Application File
------------------------------
The app.py file is the core of the application and contains:

1. MODEL LOADING:
   - Loads pre-trained YOLOv7 model from 'models/yolov7_object_detector.pt'
   - Loads pre-trained ResNet50 model from 'models/resnet50_scene_classifier.pth'
   - Sets up GPU/CPU device detection

2. IMAGE PROCESSING PIPELINE:
   - Scene Classification: Uses ResNet50 to classify weather conditions
   - Object Detection: Uses YOLOv7 to detect and locate objects
   - Anomaly Detection: Identifies safety hazards and highlights them in red
   - Image Annotation: Draws bounding boxes with class labels and confidence scores

3. VIDEO PROCESSING:
   - Frame-by-frame analysis with smart interpolation
   - Anti-flicker technology for smooth output
   - Progress tracking and status updates

4. GRADIO INTERFACE:
   - Multiple tabs for different input methods (Image, Video, Real-time)
   - Detection history and settings management
   - Download functionality for processed media

5. LOGGING SYSTEM:
   - Records all detections with timestamps
   - Tracks anomalies for safety analysis
   - Exportable detection logs in CSV format

YOLOv7 Folder Structure and Usage
=================================
The yolov7/ folder contains the complete YOLOv7 implementation:

KEY FILES:
----------
- detect.py: Standalone detection script
- train.py: Model training script
- test.py: Model evaluation script
- export.py: Model export to different formats (ONNX, TensorRT, etc.)
- hubconf.py: PyTorch Hub configuration

CONFIGURATION:
--------------
- cfg/deploy/: Model architecture configurations for deployment
- cfg/training/: Model architecture configurations for training
- data/: Dataset configurations and hyperparameters

UTILITIES:
----------
- utils/: Core utilities for datasets, plotting, metrics, etc.
- models/: Model architecture definitions
- inference/images/: Sample test images

The app.py imports key functions from yolov7/utils/ and yolov7/models/ to:
- Load the YOLOv7 model using attempt_load()
- Perform non-maximum suppression using non_max_suppression()
- Draw bounding boxes using plot_one_box()
- Preprocess images using letterbox()

SYSTEM REQUIREMENTS
===================
- Python Version: 3.11.13 (Required)
- Operating System: Windows/Linux/MacOS
- GPU: NVIDIA GPU with CUDA support (recommended) or CPU
- RAM: Minimum 8GB, Recommended 16GB+
- Storage: At least 5GB free space

INSTALLATION GUIDE
==================

Step 1: Check Python Version
-----------------------------
Ensure you have Python 3.11.13 installed:
> python --version

If you don't have Python 3.11.13, download it from:
https://www.python.org/downloads/release/python-31113/

Step 2: Clone/Download Project
------------------------------
Download this project folder to your local machine and navigate to it:
> cd path/to/Adverse_Weather_Gradio_App

Step 3: Install Required Packages
----------------------------------
Run the following command to install all dependencies:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 gradio numpy opencv-python Pillow pyyaml scipy tqdm pandas seaborn matplotlib

This command installs:
- torch, torchvision, torchaudio: PyTorch with CUDA 12.1 support
- gradio: Web interface framework
- numpy: Numerical computing
- opencv-python: Computer vision operations
- Pillow: Image processing
- pyyaml: YAML file parsing
- scipy: Scientific computing
- tqdm: Progress bars
- pandas: Data manipulation
- seaborn, matplotlib: Data visualization

Step 4: Verify Model Files
---------------------------
Ensure these model files exist in the models/ folder:
- models/yolov7_object_detector.pt
- models/resnet50_scene_classifier.pth

If missing, you'll need to obtain these pre-trained models.

Step 5: Test GPU Support (Optional)
------------------------------------
The application will automatically detect and use GPU if available.
To test GPU support, run:
> python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

RUNNING THE APPLICATION
=======================

Step 1: Launch the Application
-------------------------------
Navigate to the project folder and run:
> python app.py

Step 2: Access the Interface
----------------------------
The application will start and display:
"Running on local URL: http://127.0.0.1:9191"

Open your web browser and go to: http://127.0.0.1:9191

Step 3: Using the Interface
---------------------------

IMAGE TAB:
- Click "Upload Image" to select an image file
- Click "Analyze Image" to process
- View results: processed image, scene classification, detection log
- Download processed image using the download link

VIDEO TAB:
- Click "Upload Video File" to select a video
- Processing starts automatically (may take time)
- Download processed video when complete

REAL-TIME VIDEO TAB:
- Allow camera access when prompted
- Click "Start Detection" to begin live processing
- Toggle "Simple Mode" for better performance
- Click "Stop Detection" to end

DETECTION HISTORY TAB:
- View recent detection logs
- Save detection history to CSV file
- View color legend for object classes

TROUBLESHOOTING
===============

Common Issues and Solutions:
---------------------------

1. "CUDA not available" warning:
   - This is normal if you don't have an NVIDIA GPU
   - The app will use CPU (slower but functional)

2. Model loading errors:
   - Ensure model files exist in models/ folder
   - Check file permissions

3. Camera not working in Real-time tab:
   - Allow camera access in browser settings
   - Try refreshing the page
   - Check if camera is being used by another application

4. Slow video processing:
   - Use shorter videos for testing
   - Consider using GPU for faster processing
   - Lower video resolution if necessary

5. Out of memory errors:
   - Reduce batch size by processing fewer frames
   - Close other applications to free RAM
   - Use CPU instead of GPU if GPU memory is limited

PERFORMANCE OPTIMIZATION
========================

For Better Performance:
-----------------------
- Use NVIDIA GPU with CUDA support
- Ensure sufficient RAM (16GB+ recommended)
- Close unnecessary applications
- Use SSD storage for faster file access
- Process smaller videos initially

Real-time Detection Tips:
------------------------
- Enable "Simple Mode" for smoother real-time processing
- Good lighting improves detection accuracy
- Stable camera positioning reduces processing load

OUTPUT FILES
============

The application creates a 'gradio_outputs/' folder containing:
- Processed images with scene classification in filename
- Processed videos with timestamp
- Detection log CSV files

File naming format:
- Images: processed_[scene]_[timestamp].jpg
- Videos: processed_video_[timestamp].mp4
- Logs: detection_log_[timestamp].csv

TECHNICAL DETAILS
=================

Model Information:
-----------------
- YOLOv7: State-of-the-art object detection
- ResNet50: Robust image classification
- Input resolution: 640x640 for YOLOv7, 224x224 for ResNet50
- Detection classes: 80 COCO classes
- Scene classes: 7 weather conditions

Performance Metrics:
-------------------
- Detection confidence threshold: 0.25
- NMS IoU threshold: 0.45
- Processing speed: Depends on hardware and input size

SUPPORT AND CONTACT
===================

For technical issues or questions:
1. Check the troubleshooting section above
2. Verify all installation steps were followed correctly
3. Ensure model files are present and accessible
4. Check system requirements compatibility

Project Structure Summary:
-------------------------
├── app.py                          # Main application file
├── README.txt                      # This file
├── models/                         # Pre-trained model files
│   ├── yolov7_object_detector.pt
│   └── resnet50_scene_classifier.pth
├── yolov7/                         # YOLOv7 implementation
│   ├── detect.py                   # Detection script
│   ├── models/                     # Model architectures
│   ├── utils/                      # Utility functions
│   └── cfg/                        # Configuration files
└── gradio_outputs/                 # Generated output files

==========================================================================
                           END OF README
==========================================================================

Last Updated: July 2025
Version: 1.0
Compatible with: Python 3.11.13, CUDA 12.1, PyTorch 2.x
