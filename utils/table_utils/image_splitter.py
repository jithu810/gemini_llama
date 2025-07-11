import cv2
import numpy as np
import os

def find_split_lines(image, max_parts=3, min_space=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary = 255 - binary  # Invert so text is black (0)
    row_sums = np.sum(binary, axis=1)

    height = image.shape[0]
    target_height = height // max_parts

    cut_points = []
    current_y = target_height

    print(f"📏 Image height: {height} — Target slice height: {target_height}")

    while len(cut_points) < max_parts - 1:
        window = row_sums[current_y - min_space: current_y + min_space]
        min_val_idx = np.argmin(window)
        cut_y = current_y - min_space + min_val_idx

        if cut_y - (cut_points[-1] if cut_points else 0) < min_space:
            break  # Avoid small overlapping slices

        cut_points.append(cut_y)
        current_y = cut_y + target_height

    print(f" Suggested cut points (safe zones between text): {cut_points}")
    return cut_points

def split_image_preserving_text(image_path, output_dir="sliced_images", max_parts=8):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = cv2.imread(image_path)
    cut_points = find_split_lines(image, max_parts=max_parts)

    prev_y = 0
    saved_paths = []

    for idx, cut_y in enumerate(cut_points + [image.shape[0]]):
        slice_img = image[prev_y:cut_y, :]
        if slice_img.shape[0] > 20:
            out_path = os.path.join(output_dir, f"part_{idx + 1}.png")
            cv2.imwrite(out_path, slice_img)
            print(f"💾 Saved: {out_path} (h={slice_img.shape[0]})")
            saved_paths.append(out_path)
        else:
            print(f"⚠️ Skipped part_{idx + 1}: too small (h={slice_img.shape[0]})")
        prev_y = cut_y

    return saved_paths