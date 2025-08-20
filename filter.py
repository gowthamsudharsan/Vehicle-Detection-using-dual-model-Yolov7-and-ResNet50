import os
import shutil
import random
import threading
import time
import sys

def marquee_text(stop_event):
    msg = ">>> Filtering files... Please wait <<<   "
    while not stop_event.is_set():
        for i in range(len(msg)):
            if stop_event.is_set():
                break
            sys.stdout.write('\r' + msg[i:] + msg[:i])
            sys.stdout.flush()
            time.sleep(0.1)
    # Clear line after done
    sys.stdout.write('\r' + ' ' * len(msg) + '\r')
    sys.stdout.flush()

# --- Set seed for reproducibility ---
random.seed(42)

# --- Modify these paths ---
image_root = r"images100k"
label_root = r"labels100k"

output_image_root = r"images10k"
output_label_root = r"labels10k"

splits = ["train", "val", "test"]
total_sample_size = 10000

# --- Step 1: Collect all image paths ---
all_images = []

for split in splits:
    split_dir = os.path.join(image_root, split)
    files = [
        f for f in os.listdir(split_dir) if f.lower().endswith(".jpg")
    ]
    for f in files:
        base = os.path.splitext(f)[0]
        full_path = os.path.join(split_dir, f)
        all_images.append((split, base, full_path))

print(f"Total images available: {len(all_images)}")

# --- Step 2: Sample 10k images ---
sampled = random.sample(all_images, total_sample_size)
print(f"Sampled {len(sampled)} images.\n")

# --- Step 3: Create output folders ---
for split in splits:
    os.makedirs(os.path.join(output_image_root, split), exist_ok=True)
    os.makedirs(os.path.join(output_label_root, split), exist_ok=True)

# --- Step 4: Start marquee thread ---
stop_event = threading.Event()
marquee_thread = threading.Thread(target=marquee_text, args=(stop_event,))
marquee_thread.start()

# --- Step 5: Copy files ---
count = 0
for split, base_name, img_path in sampled:
    json_filename = base_name + ".json"
    label_path = os.path.join(label_root, split, json_filename)

    output_img_path = os.path.join(output_image_root, split, base_name + ".jpg")
    output_label_path = os.path.join(output_label_root, split, json_filename)

    shutil.copyfile(img_path, output_img_path)

    if os.path.exists(label_path):
        shutil.copyfile(label_path, output_label_path)
    else:
        print(f"\nWarning: Label not found for {base_name}")

    count += 1

# --- Step 6: Stop marquee and finish ---
stop_event.set()
marquee_thread.join()

print("Done! filtered..", count, "images and labels.")
