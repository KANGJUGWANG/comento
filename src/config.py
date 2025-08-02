# config.py
import re
import os

def get_project_root(project_folder_name="comento"):
    current = os.path.abspath(__file__)
    while True:
        parent = os.path.dirname(current)
        if os.path.basename(parent) == project_folder_name:
            return parent
        if parent == current:
            raise RuntimeError(f"❌ '{project_folder_name}' 폴더를 찾을 수 없습니다.")
        current = parent

# 🔽 YOLO 모델 자동 최신 경로 설정
def get_latest_yolo_model(base_dir):
    """
    base_dir: 'src/runs/detect/' 등 YOLO 모델 학습 결과 폴더 상위 경로
    return: 최신 'train_2nd_*' 폴더 안의 best.pt 경로
    """
    target_dir = os.path.join(base_dir, "detect")
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"YOLO detect 폴더를 찾을 수 없습니다: {target_dir}")

    candidates = [
        f for f in os.listdir(target_dir)
        if os.path.isdir(os.path.join(target_dir, f)) and re.match(r"train_2nd_\d{8}_\d{4}", f)
    ]

    if not candidates:
        raise FileNotFoundError("📁 'train_2nd_YYYYMMDD_HHMM' 폴더가 없습니다.")

    # 날짜 및 시간 기준 정렬
    candidates.sort(reverse=True)
    latest_folder = candidates[0]
    best_pt_path = os.path.join(target_dir, latest_folder, "weights", "best.pt")

    if not os.path.isfile(best_pt_path):
        raise FileNotFoundError(f"🚫 best.pt 파일이 없습니다: {best_pt_path}")

    return best_pt_path



# 🔽 프로젝트 루트 (/comento)
PROJECT_ROOT = get_project_root()

# 🔽 데이터 분할 관련 경로
ORIGIN_SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "origin_synthetic_data_set")
EDIT_SYNTHETIC_DIR   = os.path.join(PROJECT_ROOT, "synthetic_data_set")

ORIGIN_ACTUAL_DIR    = os.path.join(PROJECT_ROOT, "origin_actual_data_set", "content", "UA-DETRAC", "DETRAC_Upload")
EDIT_ACTUAL_DIR      = os.path.join(PROJECT_ROOT, "actual_data_set")

# 🔽 모델 학습 관련 YAML 파일 경로
SYNTHETIC_YAML_PATH  = os.path.join(EDIT_SYNTHETIC_DIR, "data_set.yaml")
ACTUAL_YAML_PATH     = os.path.join(EDIT_ACTUAL_DIR, "data_set.yaml")

# 🔽 FastAPI/추론용 YOLO 모델 경로
## 고정 경로 설정
# YOLO_MODEL_PATH      = os.path.join(PROJECT_ROOT, "src", "runs", "detect", "train", "weights", "best.pt")
## 자동 경로 설정
YOLO_RUNS_DIR = os.path.join(PROJECT_ROOT, "src", "runs")
# ✅ 필요할 때 호출해서 YOLO 모델 경로를 가져옴
def get_yolo_model_path():
    return get_latest_yolo_model(YOLO_RUNS_DIR)