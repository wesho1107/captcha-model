import json
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def visualize_annotations(json_path, image_root, limit=None, save_dir=None):
    json_path = Path(json_path).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    save_dir = Path(save_dir).expanduser().resolve() if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("r") as f:
        data = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    images = {img["id"]: img for img in data.get("images", [])}

    annotations_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        annotations_by_image[ann["image_id"]].append(ann)

    for idx, (image_id, image_info) in enumerate(images.items(), start=1):
        if limit is not None and idx > limit:
            break

        file_name = image_info["file_name"]
        image_path = image_root / file_name

        if not image_path.exists():
            print(f"Skipping {file_name} (missing file)")
            continue

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        for ann in annotations_by_image.get(image_id, []):
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            label_char = categories.get(ann["category_id"], f"?{ann['category_id']}")
            font = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", 16)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1 - 16), label_char, fill="blue", font=font)

        if save_dir:
            out_path = save_dir / file_name
            image.save(out_path)
        else:
            plt.figure(figsize=(6, 2))
            plt.imshow(image)
            plt.axis("off")
            plt.title(file_name)
            plt.show()

# Example call – update image_root (and maybe save_dir) before running:
visualize_annotations(
    json_path="/Users/wesho/repos/captcha-model/output.json",
    image_root="/Users/wesho/repos/captcha-model/data/train",  # update this path
    limit=20,           # set to an int to only preview first N images
    save_dir="/Users/wesho/repos/captcha-model/annotated_images"         # set to a path to save annotated copies instead of showing them
)