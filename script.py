from image_analyzer import ImageAnalyzer
import os

dataset_root = os.path.join(os.path.dirname(__file__), "dataset")

analyzer = ImageAnalyzer(dataset_root)

dataset_path_dblurred = os.path.join(dataset_root, "defocused_blurred")
dataset_path_mblurred = os.path.join(dataset_root, "motion_blurred")

image_files_dblurred = [f for f in os.listdir(dataset_path_dblurred) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files_mblurred = [f for f in os.listdir(dataset_path_mblurred) if f.endswith(('.png', '.jpg', '.jpeg'))]

analyzer.process_images(image_files_dblurred, dataset_path_dblurred, "_F", "defocused_vs_sharp.xlsx")
analyzer.process_images(image_files_mblurred, dataset_path_mblurred, "_M", "motion_vs_sharp.xlsx")