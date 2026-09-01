"""GPU kullanilabilirligini tespit eden ve her kutuphane (lgb/cat/xgb) icin
GPU denemesinin sonucunu (basarili/basarisiz) bellekte tutan yardimcilar.

Neden onbellekleniyor: orn. Optuna 50 trial x 5 fold x 3 model calisirken,
GPU gercekten calismiyorsa (orn. LightGBM standart pip kurulumu CPU-only'dir)
her fit cagrisinda GPU'yu deneyip basarisiz olmak yuzlerce kez bosuna zaman
kaybettirir. Ilk basarisizliktan sonra o kutuphane icin dogrudan CPU'ya gecilir.
"""
import shutil
import subprocess

_gpu_status = {"lgb": None, "cat": None, "xgb": None}  # None=henuz denenmedi, True/False=sonuc


def has_nvidia_gpu():
    """nvidia-smi calisiyorsa NVIDIA GPU + surucu kurulu demektir."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


GPU_AVAILABLE = has_nvidia_gpu()


def gpu_status(model_key):
    return _gpu_status.get(model_key)


def mark_gpu_result(model_key, works):
    _gpu_status[model_key] = works


def should_try_gpu(model_key, use_gpu=True):
    """Bu cagirida GPU denenmeli mi? Daha once basarisiz olduguysa hayir."""
    return use_gpu and GPU_AVAILABLE and gpu_status(model_key) is not False


# --- Kutuphaneye ozel GPU parametreleri ---

def lgb_device_params():
    # LightGBM >= 4.x, CUDA destekli ozel derleme gerektirir. Standart
    # `pip install lightgbm` CPU-only'dir -- bu yuzden bu genelde basarisiz
    # olup CPU'ya duser, bu beklenen bir durumdur.
    return {"device_type": "cuda"}


def cat_device_params():
    # CatBoost'un pip paketi GPU destegiyle gelir, surucu varsa calisir.
    return {"task_type": "GPU", "devices": "0"}


def xgb_device_params():
    # XGBoost'un pip paketi de GPU destegiyle gelir.
    return {"tree_method": "hist", "device": "cuda"}
