import kagglehub
import shutil
import os
import zipfile

# Step 1: Download the dataset via KaggleHub (goes to its cache)
path = kagglehub.dataset_download("hoangxuanviet/human-head-detection-openvm-c270")
print("Original KaggleHub cache path:", path)

# Step 2: Define destination (current working directory)
destination = "./datasets/human-head-detection-openvm"

# Step 3: Copy dataset contents to the current directory
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(destination, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print("✅ Dataset copied to current directory.")

# Step 4: Automatically unzip any .zip files found
for file in os.listdir(destination):
    if file.endswith(".zip"):
        zip_path = os.path.join(destination, file)
        extract_dir = os.path.join(destination, os.path.splitext(file)[0])
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"✅ Unzipped: {file} → {extract_dir}")

print("🎉 All dataset files ready in current directory.")