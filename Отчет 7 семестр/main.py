import os
import tempfile
import numpy as np
import open3d as o3d
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from neural_inference import NeuralReconstructionPipeline

app = FastAPI(title="3D Reconstruction API", version="1.0.0")
pipeline = NeuralReconstructionPipeline()

class SystemStatus(BaseModel):
    status: str = Field(..., description="Текущее состояние сервера")
    inference_device: str = Field(..., description="Устройство для тензорных вычислений")
    model_loaded: bool = Field(..., description="Статус загрузки весов сети")


class ReconstructionResponse(BaseModel):
    message: str = Field(..., description="Результат операции")
    processing_time_ms: int = Field(..., description="Время инференса в мс")

def simulate_multi_hole_occlusion(points, min_holes=1, max_holes=4,
                                  min_drop_per_hole=0.10, max_drop_per_hole=0.30,
                                  occlusion_prob=0.5):
    num_points = points.shape[0]

    if np.random.rand() > occlusion_prob:
        keep_ratio = np.random.uniform(0.05, 0.15)
        keep_count = int(num_points * keep_ratio)
        indices = np.random.choice(num_points, keep_count, replace=False)
        return points[indices]

    else:
        is_hole = np.zeros(num_points, dtype=bool)
        num_holes = np.random.randint(min_holes, max_holes + 1)

        for _ in range(num_holes):
            center_idx = np.random.randint(num_points)
            center_point = points[center_idx]

            scale = np.random.uniform(0.3, 2.5, size=3)
            diff = (points - center_point) * scale
            base_dist = np.linalg.norm(diff, axis=1)

            noise = np.random.uniform(0.6, 1.4, size=num_points)
            chaotic_dist = base_dist * noise

            drop_ratio = np.random.uniform(min_drop_per_hole, max_drop_per_hole)
            drop_count = int(num_points * drop_ratio)

            closest_indices = np.argsort(chaotic_dist)[:drop_count]

            is_hole[closest_indices] = True

        final_mask = np.zeros(num_points, dtype=bool)

        normal_keep_ratio = np.random.uniform(0.10, 0.25)
        normal_mask = np.random.rand(num_points) < normal_keep_ratio
        final_mask[~is_hole] = normal_mask[~is_hole]

        hole_keep_ratio = np.random.uniform(0.01, 0.04)
        hole_mask = np.random.rand(num_points) < hole_keep_ratio
        final_mask[is_hole] = hole_mask[is_hole]

        min_remaining = int(num_points * 0.03)
        remaining_count = np.sum(final_mask)

        if remaining_count < min_remaining:
            invisible_indices = np.where(~final_mask)[0]
            restore_count = min_remaining - remaining_count
            restore_indices = np.random.choice(invisible_indices, restore_count, replace=False)
            final_mask[restore_indices] = True

        return points[final_mask]

@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    return SystemStatus(status="active", inference_device=pipeline.device, model_loaded=pipeline.is_ready)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/generate_partial")
async def generate_partial(
        file: UploadFile = File(...),
        mode: str = Form("defect")
):
    if not file.filename.endswith('.obj'):
        raise HTTPException(status_code=400, detail="Формат должен быть .obj")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as temp_in:
        temp_in.write(await file.read())
        temp_in_path = temp_in.name

    try:
        mesh = o3d.io.read_triangle_mesh(temp_in_path)
        pcd = mesh.sample_points_uniformly(number_of_points=8192)
        points = np.asarray(pcd.points)

        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.linalg.norm(points, axis=1))
        points = points / max_dist

        if mode == "defect":
            final_points = simulate_multi_hole_occlusion(points)
            filename = "partial_defected.ply"
        else:
            final_points = points
            filename = "full_cloud.ply"

        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points = o3d.utility.Vector3dVector(final_points)

        temp_out_path = tempfile.mktemp(suffix=".ply")
        o3d.io.write_point_cloud(temp_out_path, final_pcd)

        return FileResponse(temp_out_path, media_type="application/octet-stream", filename=filename)
    finally:
        os.remove(temp_in_path)


@app.post("/api/reconstruct")
async def reconstruct(file: UploadFile = File(...)):
    if not file.filename.endswith('.ply'):
        raise HTTPException(status_code=400, detail="Формат должен быть .ply")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply") as temp_in:
        temp_in.write(await file.read())
        temp_in_path = temp_in.name

    try:
        result_mesh_path = pipeline.run_inference(temp_in_path)
        return FileResponse(result_mesh_path, media_type="application/octet-stream", filename="reconstructed.obj")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {str(e)}")
    finally:
        os.remove(temp_in_path)