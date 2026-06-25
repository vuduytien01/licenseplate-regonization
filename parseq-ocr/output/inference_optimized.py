import os
import sys

# ============================================================
# CRITICAL: GPU ISOLATION & NETWORK FIX FOR KAGGLE DOCKER
# ============================================================
if "LOCAL_RANK" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(os.environ["LOCAL_RANK"])
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["NCCL_P2P_DISABLE"] = "1"

sys.path.append("/kaggle/working/parseq")

import glob
import cv2
import torch
import pandas as pd
import torch.distributed as dist
import concurrent.futures

from tqdm.auto import tqdm
from ultralytics import YOLO

# ============================================================
# CẤU HÌNH TỐI ƯU CUDA
# ============================================================
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# ============================================================
# ĐƯỜNG DẪN DỮ LIỆU
# ============================================================
DATASET_DIR = "/kaggle/input/datasets/vdt1501/license-plate-detection-dataset/License Plate Detection Dataset/images/test"
YOLO_WEIGHT = "/kaggle/input/datasets/vdt1501/yolo-parseq/best.pt"
PARSEQ_WEIGHT = "/kaggle/input/datasets/vdt1501/parseq-bb5792a6-pt/parseq-bb5792a6.pt"
OUTPUT_CSV = "/kaggle/working/license_plate_results.csv"

from strhub.models.utils import create_model


# ============================================================
# HÀM HỖ TRỢ
# ============================================================
def setup_ddp():
    """Khởi tạo môi trường Distributed Data Parallel."""
    dist.init_process_group(backend="gloo")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(0) 
    return local_rank, rank, world_size


def load_image(path):
    """Đọc ảnh bằng cv2 (được gọi đa luồng)."""
    return path, cv2.imread(path)


def fast_preprocess(crop):
    """Tiền xử lý ảnh cắt (crop) trực tiếp bằng cv2 + numpy."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (128, 32), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 127.5 - 1.0
    return tensor


def batch_ocr(crops, parseq_model, device, ocr_batch_size):
    """Thực hiện OCR với CPU đa luồng cho preprocessing + FP16 cho inference."""
    tensors = []

    if len(crops) > 0:
        num_workers = min(os.cpu_count() or 4, len(crops))
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            tensors = list(executor.map(fast_preprocess, crops))

    texts_all = []
    for i in range(0, len(tensors), ocr_batch_size):
        batch = torch.stack(tensors[i:i + ocr_batch_size]).to(device, non_blocking=True)

        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = parseq_model(batch)
                probs = logits.softmax(-1)

        texts, _ = parseq_model.tokenizer.decode(probs)
        texts_all.extend(texts)

    return texts_all


# ============================================================
# HÀM CHÍNH
# ============================================================
def main():
    local_rank, rank, world_size = setup_ddp()
    device = torch.device("cuda:0")
    
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    num_cpu_cores = os.cpu_count() or 4

    # --------------------------------------------------------
    # 1. CẤU HÌNH BATCH SIZE & IMAGE SIZE
    # --------------------------------------------------------
    if vram_gb > 20:  # GPU 30GB
        YOLO_BATCH_LOCAL = 128
        YOLO_IMGSZ = 640
        OCR_BATCH_LOCAL = 8192
    else:             # GPU T4 15GB
        # GIẢM IMGSZ TỪ 640 XUỐNG 416 ĐỂ TIẾT KIỆM ~50% VRAM
        # Batch 16 ở imgsz 416 chỉ tốn ~4-5 GB VRAM
        YOLO_BATCH_LOCAL = 16
        YOLO_IMGSZ = 416
        OCR_BATCH_LOCAL = 4096

    if rank == 0:
        print("=" * 60)
        print(f"[MASTER] World size: {world_size} GPUs")
        print(f"[MASTER] CPU cores : {num_cpu_cores}")
        print("=" * 60)
    print(f"[GPU {local_rank}] {gpu_name} | VRAM: {vram_gb:.1f} GB | "
          f"YOLO_BATCH={YOLO_BATCH_LOCAL} | IMGSZ={YOLO_IMGSZ} | OCR_BATCH={OCR_BATCH_LOCAL}")

    # --------------------------------------------------------
    # 2. LOAD MODELS
    # --------------------------------------------------------
    detector = YOLO(YOLO_WEIGHT)

    parseq = create_model("parseq", pretrained=False)
    state_dict = torch.load(PARSEQ_WEIGHT, map_location="cpu")
    state_dict = {"model." + k: v for k, v in state_dict.items()}
    parseq.load_state_dict(state_dict)
    parseq = parseq.to(device).eval()

    # --------------------------------------------------------
    # 3. PHÂN VÙNG DỮ LIỆU
    # --------------------------------------------------------
    image_files = sorted([
        os.path.join(DATASET_DIR, f)
        for f in os.listdir(DATASET_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    my_files = image_files[rank::world_size]

    if rank == 0:
        print(f"[MASTER] Total images: {len(image_files)} | "
              f"Per GPU: ~{len(my_files)}")

    # --------------------------------------------------------
    # 4. ĐỌC ẢNH VÀO RAM BẰNG CPU ĐA LUỒNG
    # --------------------------------------------------------
    image_cache = {}
    if len(my_files) > 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpu_cores) as executor:
            for path, img in executor.map(load_image, my_files):
                image_cache[path] = img

    # --------------------------------------------------------
    # 5. YOLO DETECTION
    # --------------------------------------------------------
    torch.cuda.empty_cache()
    
    preds = detector.predict(
        source=my_files,
        imgsz=YOLO_IMGSZ,  # SỬ DỤNG IMGSZ ĐỘNG (416 cho T4, 640 cho GPU 30GB)
        conf=0.20,
        iou=0.50,
        batch=YOLO_BATCH_LOCAL,
        device=0,
        half=True,
        verbose=(rank == 0),
        stream=False
    )

    # --------------------------------------------------------
    # 6. OCR LOOP
    # --------------------------------------------------------
    results_all = []
    iterator = list(zip(my_files, preds))
    if rank == 0:
        iterator = tqdm(iterator, desc=f"[GPU {local_rank}] OCR")

    for img_path, pred in iterator:
        image = image_cache.get(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]
        boxes = pred.boxes.xyxy.cpu().numpy()
        
        # Scale boxes về kích thước gốc nếu imgsz != original size
        # Ultralytics tự động xử lý việc này trong pred.boxes
        
        if len(boxes) == 0:
            continue

        crops, coords = [], []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            crops.append(image[y1:y2, x1:x2])
            coords.append([x1, y1, x2, y2])

        try:
            texts = batch_ocr(crops, parseq, device, OCR_BATCH_LOCAL)
        except Exception as e:
            if rank == 0:
                print(f"[ERROR] OCR failed: {e}")
            continue

        for coord, text in zip(coords, texts):
            results_all.append({
                "image": os.path.basename(img_path),
                "plate": text,
                "bbox": coord
            })

    # --------------------------------------------------------
    # 7. LƯU KẾT QUẢ
    # --------------------------------------------------------
    out_csv_rank = OUTPUT_CSV.replace(".csv", f"_part{rank}.csv")
    pd.DataFrame(results_all).to_csv(out_csv_rank, index=False)
    print(f"[GPU {local_rank}] Saved {len(results_all)} plates -> {out_csv_rank}")

    # --------------------------------------------------------
    # 8. ĐỒNG BỘ VÀ GỘP KẾT QUẢ
    # --------------------------------------------------------
    dist.barrier()

    if rank == 0:
        part_files = sorted(glob.glob(OUTPUT_CSV.replace(".csv", "_part*.csv")))
        if part_files:
            df_combined = pd.concat(
                [pd.read_csv(f) for f in part_files],
                ignore_index=True
            )
            df_combined.to_csv(OUTPUT_CSV, index=False)
            print("=" * 60)
            print(f"[MASTER] Final CSV: {OUTPUT_CSV}")
            print(f"[MASTER] Total plates: {len(df_combined)}")
            print(f"[MASTER] File size: {os.path.getsize(OUTPUT_CSV) / 1024:.1f} KB")
            print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
