import os
import glob
import numpy as np
import open3d as o3d
import random
import tempfile
from main import simulate_multi_hole_occlusion

RAW_DIR = r"Путь до папки с .obj"
DATASET_DIR = r"Путь до папки результатов"

ORIGINAL_DIR = os.path.join(DATASET_DIR, "full")
DEFECTED_DIR = os.path.join(DATASET_DIR, "defect")
OBJ_DIR = os.path.join(DATASET_DIR, "mesh")

os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(DEFECTED_DIR, exist_ok=True)
os.makedirs(OBJ_DIR, exist_ok=True)

TARGET_TOTAL_PAIRS = 2000

def load_off_safe(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    if not lines:
        return o3d.geometry.TriangleMesh()

    first_line = lines[0].strip()
    if first_line.startswith("OFF") and len(first_line) > 3:
        lines[0] = "OFF\n" + first_line[3:] + "\n"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".off", mode='w', encoding='utf-8') as tmp:
            tmp.writelines(lines)
            tmp_path = tmp.name

        mesh = o3d.io.read_triangle_mesh(tmp_path)
        os.remove(tmp_path)
        return mesh

    return o3d.io.read_triangle_mesh(file_path)

def process_and_save(file_path, category_name, index):
    mesh = load_off_safe(file_path)

    if not mesh.has_vertices():
        return False

    pcd = mesh.sample_points_uniformly(number_of_points=8192)
    points = np.asarray(pcd.points)

    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))

    if max_dist == 0:
        return False

    points = points / max_dist

    vertices = np.asarray(mesh.vertices)
    vertices = (vertices - centroid) / max_dist
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    full_filename = f"{category_name}_{index}_full.ply"
    defect_filename = f"{category_name}_{index}_defect.ply"
    obj_filename = f"{category_name}_{index}.obj"

    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(os.path.join(ORIGINAL_DIR, full_filename), original_pcd)

    defected_points = simulate_multi_hole_occlusion(points)
    defected_pcd = o3d.geometry.PointCloud()
    defected_pcd.points = o3d.utility.Vector3dVector(defected_points)
    o3d.io.write_point_cloud(os.path.join(DEFECTED_DIR, defect_filename), defected_pcd)

    o3d.io.write_triangle_mesh(os.path.join(OBJ_DIR, obj_filename), mesh)

    return True

categories = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
categories.sort()

if categories:
    models_per_category = TARGET_TOTAL_PAIRS // len(categories)

    for category in categories:
        search_path_train = os.path.join(RAW_DIR, category, "train", "*.off")
        search_path_test = os.path.join(RAW_DIR, category, "test", "*.off")

        files = glob.glob(search_path_train) + glob.glob(search_path_test)
        random.shuffle(files)

        category_processed = 0
        for file_path in files:
            if category_processed >= models_per_category:
                break

            success = process_and_save(file_path, category, category_processed + 1)
            if success:
                category_processed += 1