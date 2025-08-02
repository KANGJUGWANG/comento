from ultralytics import YOLO
import timeit
import os
import gc
import torch
import shutil
import re
import yaml
from glob import glob
from datetime import datetime
from config import SYNTHETIC_YAML_PATH, ACTUAL_YAML_PATH
def get_latest_train_folder(base_dir="runs\detect"):
    train_dirs = sorted(
        glob(os.path.join(base_dir, "train*")),
        key=os.path.getmtime,
        reverse=True
    )
    for d in train_dirs:
        if not os.path.basename(d).startswith("train_"):
            return d
    return None

def is_interrupted_training(train_dir):
    args_path = os.path.join(train_dir, "args.yaml")
    if not os.path.exists(args_path):
        return False
    with open(args_path, 'r') as f:
        args = yaml.safe_load(f)
    return args.get("epoch", 0) < args.get("epochs", 0)

def get_latest_1st_stage_best(base_dir="runs\detect"):
    candidate_dirs = [
        os.path.join(base_dir, d) for d in os.listdir(base_dir)
        if d.startswith("train_1st_") and os.path.isdir(os.path.join(base_dir, d))
    ]
    if not candidate_dirs:
        raise FileNotFoundError("❌ 'train_1st_'로 시작하는 폴더가 없습니다.")
    latest_dir = max(candidate_dirs, key=os.path.getmtime)
    best_pt_path = os.path.join(latest_dir, "weights", "best.pt")
    if not os.path.isfile(best_pt_path):
        raise FileNotFoundError(f"❌ best.pt 파일이 없습니다: {best_pt_path}")
    return best_pt_path

def rename_latest_train_dir(base_dir="runs\detect", stage="1st"):
    train_dirs = sorted(
        glob(os.path.join(base_dir, "train*")),
        key=os.path.getmtime,
        reverse=True
    )
    if not train_dirs:
        raise FileNotFoundError(f"{base_dir} 안에 train 디렉토리가 없습니다.")
    latest_dir = train_dirs[0]
    time_suffix = datetime.now().strftime("%Y%m%d_%H%M")
    new_name = f"train_{stage}_{time_suffix}"
    new_path = os.path.join(base_dir, new_name)
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
    os.rename(latest_dir, new_path)
    print(f"✅ 폴더명 변경: '{os.path.basename(latest_dir)}' → '{new_name}'")
    return new_path

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

def train_yolo_model(yaml_path, model_weight=None, stage="1st", resume=False):
    print(f"\n🚀 [{stage}] YOLO 모델 학습 시작 ({'resume' if resume else 'new'})")
    model = YOLO(model_weight if model_weight else "yolo11s.yaml")
    start = timeit.default_timer()
    results = model.train(data=yaml_path, epochs=200, patience=10, imgsz=640, batch=-1, workers=12, resume=resume)
    end = timeit.default_timer()
    print(f"⏱️ [{stage}] 학습 시간: {int((end - start) / 60)}분")
    return rename_latest_train_dir(stage=stage)

def prompt_choice():
    print("📌 어떤 작업을 실행할까요?")
    print("1. 1차 학습만 진행 (합성 데이터)")
    print("2. 2차 학습만 진행 (실제 데이터)")
    print("3. 1차 → 2차 순차 학습")
    return input("입력 (1/2/3): ").strip()

def main():
    synthetic_yaml = SYNTHETIC_YAML_PATH
    actual_yaml = ACTUAL_YAML_PATH
    choice = prompt_choice()

    if choice == "1":
        train_yolo_model(yaml_path=synthetic_yaml, model_weight=None, stage="1st")

    elif choice == "2":
        print("🔍 2차 학습 - 중단된 학습을 이어서 진행할까요?")
        cont = input("입력: 1 = 이어서 진행 / 2 = 새로 시작: ").strip()

        if cont == "1":
            latest_train = get_latest_train_folder()
            if latest_train and is_interrupted_training(latest_train):
                latest_train = os.path.join(latest_train, "weights", "last.pt")
                print(f"✅ 중단된 학습을 이어서 진행합니다: {latest_train}")
                train_yolo_model(yaml_path=actual_yaml, model_weight=latest_train, stage="2nd", resume=True)
            else:
                print("⚠️ 중단된 학습이 없거나 이어서 진행 불가. 1차 완료 모델로 재학습합니다.")
                best_path = get_latest_1st_stage_best()
                train_yolo_model(yaml_path=actual_yaml, model_weight=best_path, stage="2nd")
        else:
            best_path = get_latest_1st_stage_best()
            print(f"✅ 1차 완료 모델을 사용하여 2차 학습 시작: {best_path}")
            train_yolo_model(yaml_path=actual_yaml, model_weight=best_path, stage="2nd")

    elif choice == "3":
        # 1차 학습
        train1_path = train_yolo_model(yaml_path=synthetic_yaml, model_weight=None, stage="1st")
        clear_memory()
        # 2차 학습
        best_weight_path = get_latest_1st_stage_best()
        train_yolo_model(yaml_path=actual_yaml, model_weight=best_weight_path, stage="2nd")

    else:
        print("❌ 잘못된 입력입니다. 프로그램을 종료합니다.")

if __name__ == "__main__":
    main()
