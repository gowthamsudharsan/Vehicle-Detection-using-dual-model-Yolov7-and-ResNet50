import torch
import warnings

# Suppress specific torch.meshgrid warning
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")

def check_gpu_availability():
    """
    Checks for GPU availability using PyTorch and prints detailed information.
    """
    if torch.cuda.is_available():
        print("CUDA (GPU support) is available!")
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")

        for i in range(num_gpus):
            print(f"\n--- GPU {i} Details ---")
            print(f"  Device Name: {torch.cuda.get_device_name(i)}")
            
            properties = torch.cuda.get_device_properties(i)
            print(f"  Compute Capability: {properties.major}.{properties.minor}")
            print(f"  Total Memory: {properties.total_memory / (1024**3):.2f} GB")
            print(f"  Multiprocessor Count: {properties.multi_processor_count}")

            print(f"  Allocated Memory: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
            print(f"  Cached Memory: {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB")
            
    else:
        print("CUDA (GPU support) is NOT available.")
        print("PyTorch will use the CPU for computations.")
        print("\nPossible reasons for CUDA not being available:")
        print("- PyTorch was installed without CUDA support (check your installation command).")
        print("- NVIDIA GPU drivers are not installed or are outdated.")
        print("- CUDA Toolkit is not installed or its version is incompatible with PyTorch.")
        print("- Your GPU is too old and not supported by the PyTorch/CUDA version.")

if __name__ == "__main__":
    check_gpu_availability()

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import sys
import random
from datetime import datetime

yolo_path = os.path.join(os.getcwd(), 'yolov7')
if yolo_path not in sys.path:
    sys.path.append(yolo_path)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.datasets import letterbox


print("Loading models...")

outputs_dir = "gradio_outputs"
os.makedirs(outputs_dir, exist_ok=True)
print(f"Created outputs directory: {outputs_dir}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

RESNET_CLASS_NAMES = ['Clear-Day', 'Foggy', 'Night', 'Overcast-Day', 'Rainy', 'Sandstorm', 'Snowy'] 

# Define anomaly classes for detection
ANOMALY_CLASSES = ['person', 'traffic light', 'traffic sign', 'bicycle'] 

try:
    model_resnet = models.resnet50()
    num_ftrs = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_ftrs, len(RESNET_CLASS_NAMES))
    model_resnet.load_state_dict(torch.load('models/resnet50_scene_classifier.pth', map_location=device, weights_only=True))
    
    model_resnet = model_resnet.to(device)
    model_resnet.eval()
    print("ResNet model loaded successfully.")

except Exception as e:
    print(f"Error loading ResNet model: {e}")
    model_resnet = None

resnet_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    weights_path = 'models/yolov7_object_detector.pt'
    model_yolo = attempt_load(weights_path, map_location=device)
    model_yolo.eval()
    YOLO_CLASS_NAMES = model_yolo.names
    print("YOLOv7 model loaded successfully.")
    print(f"YOLOv7 classes: {YOLO_CLASS_NAMES}")
except Exception as e:
    print(f"Error loading YOLOv7 model: {e}")
    model_yolo = None

def generate_class_colors(num_classes):
    """Generate distinct colors for each class"""
    colors = []
    random.seed(42)
    for i in range(num_classes):
        hue = i * 137.508
        color = [int(c) for c in np.array([
            np.sin(np.radians(hue)) * 127 + 128,
            np.sin(np.radians(hue + 120)) * 127 + 128,
            np.sin(np.radians(hue + 240)) * 127 + 128
        ])]
        colors.append(tuple(color))
    return colors

if model_yolo is not None:
    CLASS_COLORS = generate_class_colors(len(YOLO_CLASS_NAMES))
else:
    CLASS_COLORS = []

detection_log = []

def log_detection(image_id, detections, scene_class, anomalies_detected=False):
    """Log detection details"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        'timestamp': timestamp,
        'image_id': image_id,
        'scene_classification': scene_class,
        'detections': detections,
        'anomalies_detected': anomalies_detected
    }
    detection_log.append(log_entry)
    
    if len(detection_log) > 100:
        detection_log.pop(0)
    
    return log_entry

def save_detection_log_to_file():
    """Save detection log to a CSV file"""
    if not detection_log:
        return "No detections to save."
    
    try:
        import csv
        filename = f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'image_id', 'scene_classification', 'num_detections', 'detection_details', 'anomalies_detected']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for entry in detection_log:
                detection_details = "; ".join([f"{det['class']}:{det['confidence']:.2f}" for det in entry['detections']])
                writer.writerow({
                    'timestamp': entry['timestamp'],
                    'image_id': entry['image_id'],
                    'scene_classification': entry['scene_classification'],
                    'num_detections': len(entry['detections']),
                    'detection_details': detection_details,
                    'anomalies_detected': entry.get('anomalies_detected', False)
                })
        
        return f"Log saved to {filename}"
    except Exception as e:
        return f"Error saving log: {str(e)}"

def get_color_legend():
    """Generate color legend for YOLO classes"""
    if not model_yolo or not CLASS_COLORS:
        return "Model not loaded"
    
    legend_text = "Class Color Legend:\n"
    for i, (class_name, color) in enumerate(zip(YOLO_CLASS_NAMES, CLASS_COLORS)):
        hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
        legend_text += f"• {class_name}: {hex_color}\n"
    
    return legend_text

def predict(input_image):
    """
    Takes an image (as a NumPy array), performs scene classification and object detection,
    and returns the annotated image, predicted scene text, and detection log.
    """
    if model_yolo is None:
        raise gr.Error("YOLOv7 model failed to load. Cannot process.")
    
    if input_image is None:
        return None, "No image provided", ""
    
    image_id = f"img_{datetime.now().strftime('%H%M%S_%f')}"
    
    if len(input_image.shape) == 3 and input_image.shape[2] == 3:
        rgb_image = input_image.copy()
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    else:
        rgb_image = input_image.copy()
        bgr_image = input_image.copy()
    
    predicted_scene = "Unknown"
    scene_confidence_text = ""
    if model_resnet is not None:
        try:
            pil_image = Image.fromarray(rgb_image)
            image_for_resnet = resnet_transforms(pil_image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model_resnet(image_for_resnet)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                predicted_scene = RESNET_CLASS_NAMES[preds[0]]
                
                top_probs, top_indices = torch.topk(probabilities, min(3, len(RESNET_CLASS_NAMES)))
                scene_confidence_text = f"Top predictions:\n"
                for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
                    scene_confidence_text += f"{i+1}. {RESNET_CLASS_NAMES[idx]}: {prob.item():.3f}\n"
                    
        except Exception as e:
            print(f"Error in scene classification: {e}")
            predicted_scene = "Classification Error"
            scene_confidence_text = f"Error: {e}"
    
    annotated_image = rgb_image.copy()
    detections = []
    anomalies_detected = False
    anomaly_objects = []
    
    try:
        img_size = 640
        img = letterbox(bgr_image, img_size, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model_yolo(img, augment=False)[0]

        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], bgr_image.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    class_idx = int(cls)
                    class_name = YOLO_CLASS_NAMES[class_idx]
                    confidence = float(conf)
                    
                    # Check for anomalies
                    if class_name in ANOMALY_CLASSES:
                        anomalies_detected = True
                        anomaly_objects.append(class_name)
                    
                    color = CLASS_COLORS[class_idx] if class_idx < len(CLASS_COLORS) else (0, 255, 0)
                    label = f'{class_name} {confidence:.2f}'
                    
                    # Use red color for anomaly objects to highlight them
                    if class_name in ANOMALY_CLASSES:
                        color = (255, 0, 0)  # Red for anomalies
                    
                    plot_box_custom_rgb(xyxy, annotated_image, label=label, color=color, line_thickness=3)
                    
                    bbox = [int(x) for x in xyxy]
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': bbox,
                        'color': color
                    })
                    
    except Exception as e:
        print(f"Error in object detection: {e}")
        pass
    
    add_scene_label_to_image(annotated_image, predicted_scene)
    
    # Add anomaly warning if detected
    if anomalies_detected:
        add_anomaly_warning_to_image(annotated_image, anomaly_objects)
    
    log_entry = log_detection(image_id, detections, predicted_scene, anomalies_detected)
    
    detection_summary = f"Image ID: {image_id}\n"
    detection_summary += f"Scene: {predicted_scene}\n"
    detection_summary += scene_confidence_text + "\n"
    detection_summary += f"Detections: {len(detections)}\n"
    
    # Add anomaly information
    if anomalies_detected:
        unique_anomalies = list(set(anomaly_objects))
        detection_summary += f"ANOMALIES DETECTED: {', '.join(unique_anomalies)}\n"
        detection_summary += "Potential hazards identified in adverse weather!\n"
    else:
        detection_summary += "No anomalies detected\n"
    
    detection_summary += "\nDetailed detections:\n"
    for det in detections:
        anomaly_marker = "ANOMALY " if det['class'] in ANOMALY_CLASSES else "- "
        detection_summary += f"{anomaly_marker}{det['class']}: {det['confidence']:.2f}\n"
    
    return annotated_image, predicted_scene, detection_summary


def predict_with_download(input_image):
    """
    Enhanced predict function that also saves the processed image with scene name for download
    """
    annotated_image, predicted_scene, detection_summary = predict(input_image)
    download_path = save_processed_image_with_scene_name(annotated_image, predicted_scene)
    return annotated_image, predicted_scene, detection_summary, download_path


def process_video(video_path, progress=gr.Progress()):
    """
    Process a video file frame by frame and return the processed video with anti-flicker technology.
    """
    if video_path is None:
        return None
    
    if model_yolo is None:
        raise gr.Error("YOLOv7 model failed to load. Cannot process video.")
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise gr.Error("Could not open video file.")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {width}x{height}, {fps} FPS, {total_frames} frames")
        progress(0, desc="Initializing video processing...")
        
        outputs_dir = "gradio_outputs"
        os.makedirs(outputs_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"processed_video_{timestamp}.mp4"
        output_path = os.path.join(outputs_dir, output_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_path = output_path.replace('.mp4', '.avi')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise gr.Error("Could not create output video writer with any codec.")
        
        print(f"Writing output to: {output_path}")
        
        frame_count = 0
        processed_count = 0
        
        process_interval = max(1, fps // 8)
        last_processed_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % 10 == 0:
                progress_pct = frame_count / total_frames if total_frames > 0 else 0
                progress(progress_pct, desc=f"Processing frame {frame_count}/{total_frames}")
            
            if frame_count % process_interval == 0:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed_frame, _, _ = predict(frame_rgb)
                    
                    last_processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                    processed_count += 1
                    
                    out.write(last_processed_frame)
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    out.write(frame)
            else:
                if last_processed_frame is not None:
                    alpha = 0.8
                    beta = 0.2
                    
                    try:
                        blended_frame = cv2.addWeighted(last_processed_frame, alpha, frame, beta, 0)
                        out.write(blended_frame)
                    except:
                        out.write(last_processed_frame)
                else:
                    out.write(frame)
            
            frame_count += 1
        
        cap.release()
        out.release()
        
        progress(1.0, desc="Video processing complete!")
        print(f"Video processing complete. Processed {processed_count} frames out of {frame_count}")
        print(f"Processing interval: every {process_interval} frames for anti-flicker")
        print(f"Output saved to: {output_path}")
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            absolute_path = os.path.abspath(output_path)
            print(f"Returning absolute path: {absolute_path}")
            return absolute_path
        else:
            raise gr.Error("Failed to create output video file.")
            
    except Exception as e:
        print(f"Video processing error: {e}")
        raise gr.Error(f"Video processing failed: {str(e)}")


pista_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    secondary_hue=gr.themes.colors.lime,
    neutral_hue=gr.themes.colors.gray,
)

demo = gr.Blocks(theme=pista_theme, title="Adverse Weather Detection")

with demo:
    gr.Markdown(
        """
        # Vehicle Detection in Adverse Weather
        Upload an image, a video, or use your webcam to see the models in action.
        The system will classify the scene and detect vehicles and other objects.
        
        **Anomaly Detection**: The system automatically identifies potential hazards (persons, traffic lights, traffic signs, bicycles) and highlights them in red.
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="numpy", label="Upload Image")
                    image_button = gr.Button("Analyze Image", variant="primary")
                with gr.Column(scale=1):
                    image_output = gr.Image(label="Processed Output")
                    image_download = gr.File(label="Download Processed Image", visible=True)
                    with gr.Row():
                        with gr.Column(scale=1):
                            scene_output_image = gr.Textbox(label="Scene Classification")
                        with gr.Column(scale=1):
                            detection_log_image = gr.Textbox(label="Detection Log", lines=5)
                    gr.Markdown("**Features:**\n- Scene classification with confidence scores\n- Object detection with bounding boxes\n- Scene label added to bottom-right corner\n- Download includes scene name in filename\n- **Anomaly Detection**: Highlights persons, traffic lights, traffic signs, and bicycles in red as potential hazards")
        
        with gr.TabItem("Video"):
            gr.Markdown("### Video Processing")
            gr.Markdown("Upload a video file to process it with object detection. **Note:** Processing may take some time depending on video length and your hardware.")
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Video File")
                    video_status = gr.Textbox(label="Processing Status", value="Waiting for video upload...", lines=3)
                with gr.Column(scale=1):
                    video_output = gr.Video(label="Processed Video Output")
                    video_download = gr.File(label="Download Processed Video", visible=True)
                    gr.Markdown("**Tips:**\n- Supported formats: MP4, AVI, MOV\n- Processing speed depends on video length\n- Smart frame interpolation reduces flickering\n- If video doesn't play above, use the download link")

        with gr.TabItem("Real-Time Video"):
            gr.Markdown("### Real-Time Video Detection")
            gr.Markdown("This tab provides live video processing with real-time detection results. Make sure to allow camera access in your browser.")
            with gr.Row():
                with gr.Column(scale=1):
                    realtime_input = gr.Image(sources=["webcam"], type="numpy", label="Live Video Feed", streaming=True, show_download_button=False)
                    with gr.Row():
                        realtime_toggle_btn = gr.Button("Start Detection", variant="primary")
                        realtime_simple_mode = gr.Checkbox(label="Simple Mode (Direct Processing)", value=True)
                        debug_mode = gr.Checkbox(label="Debug Mode", value=False)
                    gr.Markdown("**Instructions:**\n1. Click 'Start Detection' to begin/stop real-time processing\n2. Check 'Simple Mode' for direct processing (recommended)\n3. Enable 'Debug Mode' to see detailed logs\n4. Allow camera access when prompted")
                with gr.Column(scale=1):
                    realtime_output = gr.Image(label="Live Processed Output")
                    realtime_fps_display = gr.Textbox(label="Processing Info", lines=2, value="Ready to start...")
                    with gr.Row():
                        with gr.Column(scale=1):
                            scene_output_realtime = gr.Textbox(label="Scene Classification", lines=4, value="Waiting for video feed...")
                        with gr.Column(scale=1):
                            detection_log_realtime = gr.Textbox(label="Live Detection Log", lines=6, value="Waiting for video feed...")
        
        with gr.TabItem("Detection History"):
            with gr.Column():
                gr.Markdown("### Recent Detection History")
                with gr.Row():
                    refresh_button = gr.Button("Refresh History", variant="secondary")
                    save_log_button = gr.Button("Save Log to File", variant="secondary")
                    legend_button = gr.Button("Show Color Legend", variant="secondary")
                history_output = gr.Textbox(label="Detection History", lines=15, max_lines=20)
                save_status = gr.Textbox(label="Save Status", lines=2)
                color_legend = gr.Textbox(label="Class Colors", lines=10)
        
        with gr.TabItem("Settings"):
            with gr.Column():
                gr.Markdown("### Detection Settings")
                save_log_button = gr.Button("Save Detection Log", variant="primary")
                color_legend_button = gr.Button("Show Color Legend", variant="primary")
                settings_output = gr.Textbox(label="Settings Output", lines=10, max_lines=15)
    
    image_button.click(
        predict_with_download, 
        inputs=image_input, 
        outputs=[image_output, scene_output_image, detection_log_image, image_download]
    )
    
    def process_video_with_status(video_path):
        if video_path is None:
            return None, "No video uploaded.", None
        
        yield None, "Processing video... This may take a few minutes.", None
        
        try:
            output_path = process_video(video_path)
            
            if output_path and os.path.exists(output_path):
                file_size_mb = os.path.getsize(output_path) / (1024*1024)
                final_status = f"Video processing complete!\nOutput: {os.path.basename(output_path)}\nSize: {file_size_mb:.1f} MB\n\nIf video doesn't play above, use download link below."
                
                yield output_path, final_status, output_path
            else:
                yield None, "Video processing failed - no output file generated.", None
        except Exception as e:
            yield None, f"Error: {str(e)}", None
    
    video_input.change(
        process_video_with_status, 
        inputs=video_input, 
        outputs=[video_output, video_status, video_download]
    )
    
    realtime_active = gr.State(False)
    frame_count = gr.State(0)
    
    def predict_realtime(frame, is_active, count):
        if not is_active:
            return None, "Click 'Start Detection' to begin", "Real-time detection is stopped", count, "Detection stopped - Click 'Start Detection'"
        
        if frame is None:
            return None, "Awaiting video feed...", "No frame received yet.", count, "Waiting for camera..."
        
        if model_yolo is None:
            return frame, "Model Error", "YOLOv7 model failed to load. Cannot process video.", count, "Model error"
        
        try:
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                return frame, "Format Error", f"Invalid frame format: {frame.shape}", count, "Format error"
            
            count += 1
            
            if count % 2 == 0:
                start_time = datetime.now()
                
                processed_frame, scene, detection_log = predict(frame)
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                fps = 1.0 / processing_time if processing_time > 0 else 0
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                enhanced_log = f"[{timestamp}] FPS: {fps:.1f}, Processing: {processing_time:.2f}s\n" + detection_log
                
                performance_info = f"ACTIVE | Frame: {count} | FPS: {fps:.1f} | Time: {processing_time:.2f}s"
                
                return processed_frame, scene, enhanced_log, count, performance_info
            else:
                return frame, "Processing...", f"Frame {count} (skipped for performance)", count, f"ACTIVE | Frame: {count} (processing every 2nd frame)"
                
        except Exception as e:
            error_msg = f"Processing Error: {str(e)}"
            print(f"Real-time processing error: {e}")
            import traceback
            traceback.print_exc()
            return frame, "Error", error_msg, count, f"Error on frame {count}"
    
    def toggle_realtime_detection(is_active):
        if is_active:
            return False, "Start Detection", "Real-time detection stopped", 0
        else:
            return True, "Stop Detection", "Real-time detection started", 0
    
    def predict_realtime_simple(frame, is_active=True, debug=False):
        if not is_active:
            return None, "Click 'Start Detection' to begin", "Real-time detection is stopped", "STOPPED - Click 'Start Detection'"
        
        if debug:
            print(f"[DEBUG] Frame received: {frame is not None}")
            if frame is not None:
                print(f"[DEBUG] Frame shape: {frame.shape}")
        
        if frame is None:
            return None, "Awaiting video feed...", "No frame received yet.", "Waiting for camera..."
        
        if model_yolo is None:
            return frame, "Model Error", "YOLOv7 model failed to load.", "Model error"
        
        if not isinstance(frame, np.ndarray) or len(frame.shape) != 3:
            return frame, "Format Error", "Invalid frame format", "FRAME ERROR"
        
        try:
            start_time = datetime.now()
            
            if debug:
                print(f"[DEBUG] Starting frame processing at {start_time}")
            
            processed_frame, scene, detection_log = predict(frame)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            fps = 1.0 / processing_time if processing_time > 0 else 0
            
            if debug:
                print(f"[DEBUG] Processing complete. Time: {processing_time:.2f}s, FPS: {fps:.1f}")
            
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            enhanced_log = f"[{timestamp}] " + detection_log
            
            performance_info = f"LIVE | FPS: {fps:.1f} | Time: {processing_time:.2f}s"
            
            return processed_frame, scene, enhanced_log, performance_info
            
        except Exception as e:
            error_msg = f"Processing Error: {str(e)}"
            print(f"Real-time processing error: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            return frame, "Error", error_msg, "PROCESSING ERROR"
    
    def predict_realtime_combined(frame, is_active, count, simple_mode, debug=False):
        if simple_mode:
            return predict_realtime_simple(frame, is_active, debug)
        else:
            result = predict_realtime(frame, is_active, count)
            return result[:4]
    
    realtime_toggle_btn.click(
        toggle_realtime_detection,
        inputs=[realtime_active],
        outputs=[realtime_active, realtime_toggle_btn, realtime_fps_display, frame_count]
    )
    
    try:
        realtime_input.stream(
            predict_realtime_combined,
            inputs=[realtime_input, realtime_active, frame_count, realtime_simple_mode, debug_mode],
            outputs=[realtime_output, scene_output_realtime, detection_log_realtime, realtime_fps_display],
            show_progress=False,
            stream_every=0.5
        )
    except AttributeError:
        realtime_input.change(
            predict_realtime_combined,
            inputs=[realtime_input, realtime_active, frame_count, realtime_simple_mode, debug_mode],
            outputs=[realtime_output, scene_output_realtime, detection_log_realtime, realtime_fps_display],
            show_progress=False
        )
    
    def get_detection_history():
        if not detection_log:
            return "No detections recorded yet."
        
        history_text = ""
        for entry in reversed(detection_log[-10:]):
            anomaly_status = "ANOMALIES DETECTED" if entry.get('anomalies_detected', False) else "Normal"
            history_text += f"[{entry['timestamp']}] ID: {entry['image_id']} | {anomaly_status}\n"
            history_text += f"Scene: {entry['scene_classification']}\n"
            history_text += f"Detections ({len(entry['detections'])}):\n"
            for det in entry['detections']:
                anomaly_marker = "ALERT: " if det['class'] in ANOMALY_CLASSES else "  - "
                history_text += f"{anomaly_marker}{det['class']}: {det['confidence']:.2f}\n"
            history_text += "-" * 50 + "\n"
        
        return history_text
    
    refresh_button.click(get_detection_history, outputs=history_output)
    save_log_button.click(save_detection_log_to_file, outputs=save_status)
    legend_button.click(get_color_legend, outputs=color_legend)
    
    save_log_button.click(save_detection_log_to_file, outputs=settings_output)
    color_legend_button.click(get_color_legend, outputs=settings_output)

def plot_box_custom(xyxy, img, label='', color=(255, 0, 0), line_thickness=3):
    """Plot one bounding box on image img with custom color handling (BGR format)"""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def plot_box_custom_rgb(xyxy, img, label='', color=(255, 0, 0), line_thickness=3):
    """Plot one bounding box on RGB image with custom color handling"""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    color_bgr = (color[2], color[1], color[0])
    
    cv2.rectangle(img_bgr, c1, c2, color_bgr, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img_bgr, c1, c2, color_bgr, -1, cv2.LINE_AA)
        cv2.putText(img_bgr, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    
    img[:] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def add_scene_label_to_image(img, scene_text, font_scale=0.8, thickness=2):
    """
    Add scene classification text to the bottom right corner of an RGB image
    """
    try:
        h, w = img.shape[:2]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Scene: {scene_text}"
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        padding_x = 10
        padding_y = 10
        text_x = w - text_width - padding_x
        text_y = h - padding_y
        
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        rect_x1 = text_x - 5
        rect_y1 = text_y - text_height - 5
        rect_x2 = w - 5
        rect_y2 = h - 5
        
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, img_bgr)
        
        cv2.putText(img_bgr, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        img[:] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        print(f"Error adding scene label to image: {e}")

def save_processed_image_with_scene_name(image, scene_text, original_filename=None):
    """
    Save processed image with scene classification in filename
    """
    try:
        outputs_dir = "gradio_outputs"
        os.makedirs(outputs_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        clean_scene = scene_text.replace(' ', '_').replace('-', '_').lower()
        
        if original_filename:
            name, ext = os.path.splitext(original_filename)
            if not ext:
                ext = '.jpg'
            filename = f"processed_{clean_scene}_{timestamp}{ext}"
        else:
            filename = f"processed_{clean_scene}_{timestamp}.jpg"
        
        filepath = os.path.join(outputs_dir, filename)
        
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        success = cv2.imwrite(filepath, image_bgr)
        
        if success:
            return os.path.abspath(filepath)
        else:
            print(f"Failed to save image to {filepath}")
            return None
            
    except Exception as e:
        print(f"Error saving processed image: {e}")
        return None

def add_anomaly_warning_to_image(img, anomaly_objects, font_scale=0.9, thickness=2):
    """
    Add anomaly warning text to the top left corner of an RGB image
    """
    try:
        h, w = img.shape[:2]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        warning_text = "ANOMALIES DETECTED!"
        details_text = f"Objects: {', '.join(set(anomaly_objects))}"
        
        # Convert RGB to BGR temporarily for OpenCV text operations
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Get text sizes
        (warning_width, warning_height), _ = cv2.getTextSize(warning_text, font, font_scale, thickness)
        (details_width, details_height), _ = cv2.getTextSize(details_text, font, font_scale-0.2, thickness-1)
        
        # Calculate position (top left with some padding)
        padding_x = 15
        padding_y = 30
        warning_x = padding_x
        warning_y = padding_y + warning_height
        details_x = padding_x
        details_y = warning_y + details_height + 10
        
        # Create background rectangle for better readability
        rect_width = max(warning_width, details_width) + 20
        rect_height = warning_height + details_height + 25
        rect_x1 = padding_x - 10
        rect_y1 = padding_y - 5
        rect_x2 = rect_x1 + rect_width
        rect_y2 = rect_y1 + rect_height
        
        # Add semi-transparent red background
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 200), -1)  # Red background
        
        # Apply the overlay with transparency
        alpha = 0.7  # Transparency factor
        cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, img_bgr)
        
        # Add warning text in white
        cv2.putText(img_bgr, warning_text, (warning_x, warning_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(img_bgr, details_text, (details_x, details_y), font, font_scale-0.2, (255, 255, 255), thickness-1, cv2.LINE_AA)
        
        # Convert back to RGB and update the original image
        img[:] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        print(f"Error adding anomaly warning to image: {e}")

print("Launching Gradio UI...")
demo.launch(
    share=False,
    server_name="127.0.0.1",
    server_port=9191,
    show_error=True,
    allowed_paths=["gradio_outputs"]
)