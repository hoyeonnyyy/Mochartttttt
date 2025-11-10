import json
import random
import shutil
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Paths
base_dir = Path("/home/khy/Project_CMU/chart_classification/ChartQA_Dataset/test")
output_dir = Path("/home/khy/Project_CMU/chart_classification/ChartQA_Dataset/test_simple")
png_dir = base_dir / "png"
output_png_dir = output_dir / "png"

# Create output directories
output_dir.mkdir(exist_ok=True)
output_png_dir.mkdir(exist_ok=True)

# Load test_human.json and test_augmented.json
with open(base_dir / "test_human.json", "r") as f:
    test_human = json.load(f)

with open(base_dir / "test_augmented.json", "r") as f:
    test_augmented = json.load(f)

print(f"Original test_human: {len(test_human)} samples")
print(f"Original test_augmented: {len(test_augmented)} samples")

# Random sample 100 from each
sampled_human = random.sample(test_human, 100)
sampled_augmented = random.sample(test_augmented, 100)

# Extract unique image filenames
human_images = set([item["imgname"] for item in sampled_human])
augmented_images = set([item["imgname"] for item in sampled_augmented])
all_images = human_images.union(augmented_images)

print(f"\nSampled 100 from test_human")
print(f"Sampled 100 from test_augmented")
print(f"Unique images to copy: {len(all_images)}")

# Copy PNG files
copied_count = 0
missing_count = 0

for img_name in all_images:
    src_path = png_dir / img_name
    dst_path = output_png_dir / img_name

    if src_path.exists():
        shutil.copy2(src_path, dst_path)
        copied_count += 1
    else:
        print(f"Warning: {img_name} not found")
        missing_count += 1

# Save sampled JSON files
with open(output_dir / "test_human.json", "w") as f:
    json.dump(sampled_human, f, indent=2)

with open(output_dir / "test_augmented.json", "w") as f:
    json.dump(sampled_augmented, f, indent=2)

print(f"\nResults:")
print(f"- Copied {copied_count} PNG files")
print(f"- Missing {missing_count} files")
print(f"- Saved test_human.json with 100 samples")
print(f"- Saved test_augmented.json with 100 samples")
print(f"\nOutput directory: {output_dir}")
