import open3d as o3d
import numpy as np
from pathlib import Path


# Симуляция камерной окклюзии
def simulate_camera_occlusion(points, normals=None, camera_direction=None,
                              cone_angle_deg=45.0, keep_outside_cone=True):

    num_points = points.shape[0]
    centroid = np.mean(points, axis=0)

    # Выбор случайного направления камеры
    if camera_direction is None:
        phi = np.random.uniform(0, 2 * np.pi)
        costheta = np.random.uniform(-1, 1)
        theta = np.arccos(costheta)
        camera_direction = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)])
    camera_direction /= np.linalg.norm(camera_direction)

    # Векторы от центра до точек, единичные направления
    vecs = points - centroid
    vecs_norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_norm = np.where(vecs_norm > 0, vecs_norm, 1.0)  # защита от нулевых векторов
    unit_vecs = vecs / vecs_norm

    # Косинус и угол между направлением на камеру и вектором точки
    cos_angles = np.dot(unit_vecs, camera_direction)
    angles_rad = np.arccos(np.clip(cos_angles, -1.0, 1.0))

    cone_angle_rad = np.deg2rad(cone_angle_deg)
    inside_cone_mask = angles_rad <= cone_angle_rad   # точки, попавшие в конус окклюзии

    # Учет видимости по нормалям
    if normals is not None:
        dot_normals = np.dot(normals, camera_direction)
        visible_by_normal = dot_normals > 0
    else:
        visible_by_normal = np.ones(num_points, dtype=bool)

    # Формирование финальной маски
    if keep_outside_cone:
        final_mask = ~inside_cone_mask & visible_by_normal
    else:
        final_mask = inside_cone_mask & visible_by_normal

    # Защита от исчезновения облака
    if np.sum(final_mask) < 10:
        return points[np.random.choice(num_points, max(10, num_points // 10), replace=False)]

    return points[final_mask]


# Семплирование полного облака
def sample_point_cloud_from_mesh(mesh_path, num_points=2048):
    try:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if len(mesh.triangles) == 0:
            return None, None

        # Равномерное семплирование
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)

        # Центрирование и масштабирование в единичную сферу
        center = pcd.get_center()
        pcd.translate(-center)
        max_dist = np.linalg.norm(np.asarray(pcd.points), axis=1).max()
        if max_dist > 0:
            pcd.scale(1.0 / max_dist, center=np.zeros(3))

        return pcd, np.asarray(pcd.points)
    except Exception as e:
        print(f" Error reading mesh {mesh_path}: {e}")
        return None, None


# Основной цикл создания базы данных
def create_dataset_from_obj(source_dir, output_dir, points_per_cloud=2048,
                            cone_angle_deg=45.0, keep_outside_cone=True):

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    complete_dir = output_path / "complete"
    partial_dir = output_path / "partial"
    complete_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)

    obj_files = list(source_path.glob("*.obj"))
    print(f"Found {len(obj_files)} .obj files")

    for i, obj_path in enumerate(obj_files, 1):
        name = obj_path.stem
        print(f"[{i}/{len(obj_files)}] Processing {name}")

        pcd_complete, pts = sample_point_cloud_from_mesh(obj_path, points_per_cloud)
        if pcd_complete is None:
            continue

        # Сохранение полного облака
        o3d.io.write_point_cloud(str(complete_dir / f"{name}_complete.ply"), pcd_complete)

        # Генерация частичного облака
        partial_pts = simulate_camera_occlusion(
            pts,
            cone_angle_deg=cone_angle_deg,
            keep_outside_cone=keep_outside_cone
        )

        pcd_partial = o3d.geometry.PointCloud()
        pcd_partial.points = o3d.utility.Vector3dVector(partial_pts)
        o3d.io.write_point_cloud(str(partial_dir / f"{name}_partial.ply"), pcd_partial)

    print(f"\nDone. Dataset saved to {output_path}")

if __name__ == "__main__":
    obj_folder = "./obj"
    output_folder = "./my_point_dataset"
    create_dataset_from_obj(obj_folder, output_folder, points_per_cloud=2048,
                            cone_angle_deg=45.0, keep_outside_cone=True)