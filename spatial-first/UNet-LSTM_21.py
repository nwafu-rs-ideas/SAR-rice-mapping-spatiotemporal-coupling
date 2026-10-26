import os
import random
import math
import datetime
import numpy as np
import rasterio
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, UpSampling2D, Lambda, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import Sequence
from tensorflow.keras import mixed_precision

from scipy.ndimage import uniform_filter
from tqdm import tqdm
import time


# ============================================================
# 基本设置
# ============================================================

mixed_precision.set_global_policy('float32')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = r"E:\WangRuozhen\RiceMapping\Code\Code_21\UNet"
LOG_DIR = r"E:\WangRuozhen\RiceMapping\logs"

# 固定好的 2021 6:2:2 数据划分
SPLIT_DIR = r"D:\WangRuozhen\RiceMapping\Data\Split_2021_seed42"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

YEAR = 2021
PATCH_SIZE = 128
STRIDE = 64
BATCH_SIZE = 10
THRESHOLD = 0.5
N_BOOTSTRAP = 1000


# +++++++++++++++++++++++++++++++++++++++++++
#     评价指标
# +++++++++++++++++++++++++++++++++++++++++++

def recall(y_true, y_pred):
    TP = K.sum(K.cast(y_true * K.round(y_pred), 'float'))
    recall_value = TP / (K.sum(K.cast(y_true, 'float')) + K.epsilon())
    return recall_value


def precision(y_true, y_pred):
    TP = K.sum(K.cast(y_true * K.round(y_pred), 'float'))
    precision_value = TP / (K.sum(K.cast(K.round(y_pred), 'float')) + K.epsilon())
    return precision_value


def fmeasure(y_true, y_pred):
    TP = K.sum(K.cast(y_true * K.round(y_pred), 'float'))
    precision_val = TP / (K.sum(K.cast(K.round(y_pred), 'float')) + K.epsilon())
    recall_val = TP / (K.sum(K.cast(K.round(y_true), 'float')) + K.epsilon())
    F1score = 2 * precision_val * recall_val / (precision_val + recall_val + K.epsilon())
    return F1score


def kappa_metrics(y_true, y_pred):
    TP = K.sum(K.cast(y_true * K.round(y_pred), 'float')) + K.epsilon()
    FP = K.sum(K.cast((1 - y_true) * K.round(y_pred), 'float')) + K.epsilon()
    FN = K.sum(K.cast(y_true * (1 - K.round(y_pred)), 'float')) + K.epsilon()
    TN = K.sum(K.cast((1 - y_true) * (1 - K.round(y_pred)), 'float')) + K.epsilon()

    totalnum = TP + FP + FN + TN
    p0 = (TP + TN) / (totalnum + K.epsilon())
    pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (totalnum * totalnum + K.epsilon())
    kappa_coef = (p0 - pe) / (1 - pe + K.epsilon())

    return kappa_coef


def OA(y_true, y_pred):
    TP = K.sum(K.cast(y_true * K.round(y_pred), 'float'))
    FP = K.sum(K.cast((1 - y_true) * K.round(y_pred), 'float'))
    FN = K.sum(K.cast(y_true * (1 - K.round(y_pred)), 'float'))
    TN = K.sum(K.cast((1 - y_true) * (1 - K.round(y_pred)), 'float'))

    totalnum = TP + FP + FN + TN
    overallAC = (TP + TN) / (totalnum + K.epsilon())

    return overallAC


def IoU(y_true, y_pred):
    y_pred = K.round(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    iou = (intersection + K.epsilon()) / (union + K.epsilon())

    return iou


# +++++++++++++++++++++++++++++++++++++++++++
#     Lee 滤波
# +++++++++++++++++++++++++++++++++++++++++++

def refined_lee_filter(data, window_size=7, cu=0.523):

    data_linear = 10 ** (data / 10)
    data_linear = np.nan_to_num(data_linear, nan=np.nanmedian(data_linear))

    mean = uniform_filter(data_linear, window_size, mode='constant', cval=np.nanmedian(data_linear))
    mean_sq = uniform_filter(data_linear ** 2, window_size, mode='constant', cval=np.nanmedian(data_linear) ** 2)
    variance = np.maximum(mean_sq - mean ** 2, 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        enl = np.zeros_like(variance)

        mask = variance > 1e-10
        enl[mask] = (mean[mask] ** 2) / variance[mask]
        enl[~mask] = 1e6

        weights = np.ones_like(variance)
        valid_mask = (enl > 1e-10) & (variance > 1e-10)
        weights[valid_mask] = 1 - (cu ** 2 / (enl[valid_mask] * variance[valid_mask]))

    weights = np.clip(weights, 0, 1)

    filtered_linear = mean + weights * (data_linear - mean)
    filtered = 10 * np.log10(np.maximum(filtered_linear, 1e-10))
    filtered = np.nan_to_num(filtered, nan=np.nanmin(filtered))

    return filtered


# +++++++++++++++++++++++++++++++++++++++++++
#     加载 SAR 数据
# +++++++++++++++++++++++++++++++++++++++++++

def load_sar_data(sar_dir, year, apply_filter=True):

    vh_files = sorted([f for f in os.listdir(sar_dir) if f.endswith('.tif') and f'VH_{year}' in f])
    vv_files = sorted([f for f in os.listdir(sar_dir) if f.endswith('.tif') and f'VV_{year}' in f])

    print(f"找到 {len(vh_files)} 个VH文件和 {len(vv_files)} 个VV文件")

    for f in vh_files + vv_files:
        file_path = os.path.join(sar_dir, f)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

    first_vh_file = os.path.join(sar_dir, vh_files[0])

    with rasterio.open(first_vh_file) as src:
        height, width = src.shape
        profile = src.profile.copy()
        print(f"图像尺寸: {height} x {width}")

    total_channels = len(vh_files) + len(vv_files)
    sar_data = np.zeros((height, width, total_channels), dtype=np.float32)

    VH_MIN, VH_MAX = -29.0, -11.0
    VV_MIN, VV_MAX = -22.0, -5.0

    n_times = len(vh_files)

    vh_data = np.zeros((height, width, n_times), dtype=np.float32)
    vv_data = np.zeros((height, width, n_times), dtype=np.float32)

    for i, f in enumerate(tqdm(vh_files, desc="处理VH数据")):
        file_path = os.path.join(sar_dir, f)

        try:
            with rasterio.open(file_path) as src:
                band = src.read(1)

                band = np.nan_to_num(band, nan=VH_MIN)

                if apply_filter:
                    band = refined_lee_filter(band)

                band = np.clip(band, VH_MIN, VH_MAX)
                normalized_band = (band - VH_MIN) / (VH_MAX - VH_MIN)

                vh_data[..., i] = normalized_band

        except Exception as e:
            print(f"处理文件 {f} 时出错: {e}")
            vh_data[..., i] = np.mean(vh_data[..., :i], axis=2) if i > 0 else 0.5

    for j, f in enumerate(tqdm(vv_files, desc="处理VV数据")):
        file_path = os.path.join(sar_dir, f)

        try:
            with rasterio.open(file_path) as src:
                band = src.read(1)

                band = np.nan_to_num(band, nan=VV_MIN)

                if apply_filter:
                    band = refined_lee_filter(band)

                band = np.clip(band, VV_MIN, VV_MAX)
                normalized_band = (band - VV_MIN) / (VV_MAX - VV_MIN)

                vv_data[..., j] = normalized_band

        except Exception as e:
            print(f"处理文件 {f} 时出错: {e}")
            vv_data[..., j] = np.mean(vv_data[..., :j], axis=2) if j > 0 else 0.5

    print("交错合并VH和VV通道...")

    for t in range(n_times):
        sar_data[..., t * 2] = vh_data[..., t]
        sar_data[..., t * 2 + 1] = vv_data[..., t]

    print(f"通道交错完成: [VH1, VV1, VH2, VV2, ..., VH{n_times}, VV{n_times}]")

    nan_count = np.sum(np.isnan(sar_data))
    inf_count = np.sum(np.isinf(sar_data))

    if nan_count > 0 or inf_count > 0:
        print(f"警告: 数据中包含 {nan_count} 个NaN值和 {inf_count} 个Inf值")
        sar_data = np.nan_to_num(sar_data, nan=0.0, posinf=1.0, neginf=0.0)

    print(f"数据加载完成，形状: {sar_data.shape}")
    print(f"数据范围: [{np.min(sar_data):.4f}, {np.max(sar_data):.4f}]")

    return sar_data, profile


# +++++++++++++++++++++++++++++++++++++++++++
#     加载标签
# +++++++++++++++++++++++++++++++++++++++++++

def load_label_data(label_path):

    with rasterio.open(label_path) as src:
        label = src.read(1)
        profile = src.profile

    return (label == 3).astype(np.uint8), profile


# +++++++++++++++++++++++++++++++++++++++++++
#     数据生成器
# +++++++++++++++++++++++++++++++++++++++++++

class SARDataGenerator(Sequence):

    def __init__(self, sar_data, label_data, coords, batch_size=10, shuffle=True, augment=True, patch_size=128):

        self.sar_data = sar_data.astype(np.float32)
        self.label_data = label_data.astype(np.float32)

        self.coords = [tuple(map(int, coord)) for coord in coords]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.patch_size = patch_size

        self.height, self.width = self.label_data.shape

        self.on_epoch_end()

    # 修改：原来是 len(self.coords) // self.batch_size
    # 现在改成 ceil，防止最后几个 patch 被丢掉
    def __len__(self):
        return math.ceil(len(self.coords) / self.batch_size)

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.coords)

    def __getitem__(self, idx):

        batch_coords = self.coords[idx * self.batch_size:(idx + 1) * self.batch_size]

        X_batch = []
        y_batch = []

        for y, x in batch_coords:

            if y + self.patch_size > self.height or x + self.patch_size > self.width:
                continue

            X = self.sar_data[y:y + self.patch_size, x:x + self.patch_size, :]

            y_patch = self.label_data[y:y + self.patch_size, x:x + self.patch_size]

            if X.shape[:2] != (self.patch_size, self.patch_size):
                continue

            if y_patch.shape != (self.patch_size, self.patch_size):
                continue

            if self.augment:
                X, y_patch = self._augment(X, y_patch)

            y_patch = np.clip(y_patch, 0, 1)
            y_patch = y_patch.astype(np.float32)
            y_patch = np.expand_dims(y_patch, axis=-1)

            X_batch.append(X)
            y_batch.append(y_patch)

        if len(X_batch) == 0:
            raise RuntimeError(f"Batch {idx} 没有有效 patch")

        X_batch = np.stack(X_batch, axis=0).astype(np.float32)
        y_batch = np.stack(y_batch, axis=0).astype(np.float32)

        assert X_batch.ndim == 4
        assert y_batch.ndim == 4
        assert X_batch.shape[0] == y_batch.shape[0]
        assert y_batch.shape[-1] == 1

        return X_batch, y_batch

    def _augment(self, X, y):

        if np.random.rand() < 0.5:
            k = np.random.randint(0, 4)
            X = np.rot90(X, k)
            y = np.rot90(y, k)

        if np.random.rand() < 0.5:
            X = np.fliplr(X)
            y = np.fliplr(y)

        if np.random.rand() < 0.5:
            X = np.flipud(X)
            y = np.flipud(y)

        return X, y


# +++++++++++++++++++++++++++++++++++++++++++
#     UNet + LSTM
#     保留原模型结构
# +++++++++++++++++++++++++++++++++++++++++++

def build_UNet_LSTM_model(input_shape=(128, 128, 18)):
    inputs = keras.layers.Input(input_shape)

    conv1 = Conv2D(32, 3, activation="relu", padding="same")(inputs)
    conv1 = Conv2D(32, 3, activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D()(conv1)

    conv2 = Conv2D(64, 3, activation="relu", padding="same")(pool1)
    conv2 = Conv2D(64, 3, activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D()(conv2)

    conv3 = Conv2D(128, 3, activation="relu", padding="same")(pool2)
    conv3 = Conv2D(128, 3, activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D()(conv3)

    conv4 = Conv2D(256, 3, activation="relu", padding="same")(pool3)
    conv4 = Conv2D(256, 3, activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D()(conv4)

    conv5 = Conv2D(512, 3, activation="relu", padding="same")(pool4)
    conv5 = Conv2D(512, 3, activation="relu", padding="same")(conv5)

    up6 = concatenate([UpSampling2D()(conv5), conv4])
    conv6 = Conv2D(256, 3, activation="relu", padding="same")(up6)
    conv6 = Conv2D(256, 3, activation="relu", padding="same")(conv6)

    up7 = concatenate([UpSampling2D()(conv6), conv3])
    conv7 = Conv2D(128, 3, activation="relu", padding="same")(up7)
    conv7 = Conv2D(128, 3, activation="relu", padding="same")(conv7)

    up8 = concatenate([UpSampling2D()(conv7), conv2])
    conv8 = Conv2D(64, 3, activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, 3, activation="relu", padding="same")(conv8)

    up9 = concatenate([UpSampling2D()(conv8), conv1])
    conv9 = Conv2D(32, 3, activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, 3, activation="relu", padding="same")(conv9)

    def flatten_pixels(x):
        b = tf.shape(x)[0]
        return tf.reshape(x, [b, 128 * 128, 32])

    seq = Lambda(flatten_pixels)(conv9)
    lstm_out = LSTM(64, return_sequences=True, dtype="float32")(seq)

    def restore_map(x):
        b = tf.shape(x)[0]
        return tf.reshape(x, [b, 128, 128, 64])

    feat_map = Lambda(restore_map)(lstm_out)
    outputs = Conv2D(1, 1, activation="sigmoid", dtype="float32")(feat_map)

    return Model(inputs=inputs, outputs=outputs, name="UNet_LSTM")


# +++++++++++++++++++++++++++++++++++++++++++
#     参数量
# +++++++++++++++++++++++++++++++++++++++++++

def quick_params_summary(model):

    total = model.count_params()
    trainable = sum([np.prod(w.shape) for w in model.trainable_weights])
    non_trainable = total - trainable

    print("📊 模型参数量统计:")
    print(f"  总计: {total:,} ≈ {total / 1e6:.2f}M")
    print(f"  可训练: {trainable:,} ≈ {trainable / 1e6:.2f}M ({trainable / total * 100:.1f}%)")
    print(f"  不可训练: {non_trainable:,} ≈ {non_trainable / 1e6:.3f}M")

    return total, trainable, non_trainable


# +++++++++++++++++++++++++++++++++++++++++++
#     论文最终指标
#     Global metric + patch bootstrap
# +++++++++++++++++++++++++++++++++++++++++++

def evaluate_global_metrics_with_patch_bootstrap(model, test_gen, threshold=0.5, n_bootstrap=1000, seed=42):

    eps = 1e-7
    patch_counts = []

    for i in tqdm(range(len(test_gen)), desc="Collect patch confusion matrix"):

        X_batch, y_batch = test_gen[i]

        y_pred_prob = model.predict(X_batch, verbose=0)
        y_pred = (y_pred_prob >= threshold).astype(np.uint8)
        y_true = y_batch.astype(np.uint8)

        for j in range(y_true.shape[0]):

            yt = y_true[j].reshape(-1)
            yp = y_pred[j].reshape(-1)

            TP = np.sum((yt == 1) & (yp == 1))
            TN = np.sum((yt == 0) & (yp == 0))
            FP = np.sum((yt == 0) & (yp == 1))
            FN = np.sum((yt == 1) & (yp == 0))

            patch_counts.append([TP, TN, FP, FN])

    patch_counts = np.array(patch_counts, dtype=np.float64)

    print("\n实际评价 patch 数:", len(patch_counts))
    print("期望测试 patch 数:", len(test_gen.coords))

    def calc_metrics_from_counts(counts):

        TP = counts[:, 0].sum()
        TN = counts[:, 1].sum()
        FP = counts[:, 2].sum()
        FN = counts[:, 3].sum()

        accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
        precision_value = TP / (TP + FP + eps)
        recall_value = TP / (TP + FN + eps)
        f1_value = 2 * precision_value * recall_value / (precision_value + recall_value + eps)

        return accuracy, precision_value, recall_value, f1_value

    overall_acc, overall_precision, overall_recall, overall_f1 = calc_metrics_from_counts(patch_counts)

    rng = np.random.default_rng(seed)
    n_patches = len(patch_counts)

    boot_acc = []
    boot_precision = []
    boot_recall = []
    boot_f1 = []

    for _ in tqdm(range(n_bootstrap), desc="Bootstrap evaluation"):

        sample_idx = rng.choice(n_patches, size=n_patches, replace=True)
        sample_counts = patch_counts[sample_idx]

        acc, pre, rec, f1 = calc_metrics_from_counts(sample_counts)

        boot_acc.append(acc)
        boot_precision.append(pre)
        boot_recall.append(rec)
        boot_f1.append(f1)

    boot_acc = np.array(boot_acc)
    boot_precision = np.array(boot_precision)
    boot_recall = np.array(boot_recall)
    boot_f1 = np.array(boot_f1)

    acc_std = boot_acc.std(ddof=1)
    precision_std = boot_precision.std(ddof=1)
    recall_std = boot_recall.std(ddof=1)
    f1_std = boot_f1.std(ddof=1)

    print("\n" + "=" * 75)
    print("Global Metrics with Patch-level Bootstrap")
    print("=" * 75)

    print(f"Accuracy : {overall_acc:.4f} ± {acc_std:.4f}")
    print(f"Precision: {overall_precision:.4f} ± {precision_std:.4f}")
    print(f"Recall   : {overall_recall:.4f} ± {recall_std:.4f}")
    print(f"F1-Score : {overall_f1:.4f} ± {f1_std:.4f}")

    print("\n论文表格格式：")
    print(
        f"{overall_acc:.4f}±{acc_std:.4f}\t"
        f"{overall_precision:.4f}±{precision_std:.4f}\t"
        f"{overall_recall:.4f}±{recall_std:.4f}\t"
        f"{overall_f1:.4f}±{f1_std:.4f}"
    )

    return {
        "Accuracy": overall_acc,
        "Accuracy_std": acc_std,
        "Precision": overall_precision,
        "Precision_std": precision_std,
        "Recall": overall_recall,
        "Recall_std": recall_std,
        "F1": overall_f1,
        "F1_std": f1_std
    }


# +++++++++++++++++++++++++++++++++++++++++++
#     保存固定 test set 的预测结果
#     后面 Fig.11 直接读这个文件
# +++++++++++++++++++++++++++++++++++++++++++

def save_test_predictions(model, test_gen, output_dir, model_tag, threshold=0.5):

    print("\n" + "=" * 75)
    print("保存固定 test set 预测和 confidence")
    print("=" * 75)

    all_prob = []
    all_true = []

    for i in tqdm(range(len(test_gen)), desc="Test prediction"):
        X_batch, y_batch = test_gen[i]

        prob_batch = model.predict(X_batch, verbose=0)

        all_prob.append(prob_batch.reshape(-1))
        all_true.append(y_batch.reshape(-1))

    prob = np.concatenate(all_prob).astype(np.float32)
    y_true = np.concatenate(all_true).astype(np.uint8)

    y_pred = (prob >= threshold).astype(np.uint8)

    correct = y_pred == y_true
    wrong = y_pred != y_true

    confidence = np.maximum(prob, 1.0 - prob)

    n_total = len(y_true)
    n_correct = int(correct.sum())
    n_wrong = int(wrong.sum())

    OA_check = n_correct / n_total

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    eps = 1e-7

    precision_value = TP / (TP + FP + eps)
    recall_value = TP / (TP + FN + eps)
    f1_value = 2 * precision_value * recall_value / (precision_value + recall_value + eps)

    print("\n固定 test set 检查：")
    print("Test patches:", len(test_gen.coords))
    print("Total pixels:", n_total)
    print("Correct pixels:", n_correct)
    print("Wrong pixels:", n_wrong)

    print(f"OA from Correct/Total: {OA_check:.6f}")
    print(f"Precision: {precision_value:.6f}")
    print(f"Recall: {recall_value:.6f}")
    print(f"F1: {f1_value:.6f}")

    print("TP:", TP)
    print("TN:", TN)
    print("FP:", FP)
    print("FN:", FN)

    # ========================================================
    # 使用与模型完全相同的名称/时间戳
    # ========================================================

    save_path = os.path.join(
        output_dir,
        f"test_prediction_{model_tag}.npz"
    )

    np.savez_compressed(
        save_path,
        prob=prob,
        y_true=y_true,
        y_pred=y_pred,
        confidence=confidence,
        correct=correct,
        wrong=wrong
    )

    print("\n✅ 固定 test set 预测结果已保存：")
    print(save_path)

    return {
        "OA": OA_check,
        "Precision": precision_value,
        "Recall": recall_value,
        "F1": f1_value,
        "Correct": n_correct,
        "Wrong": n_wrong,
        "Total": n_total
    }


# +++++++++++++++++++++++++++++++++++++++++++
#     UNet + LSTM
#     2021 train / 2021 val / 2021 test
# +++++++++++++++++++++++++++++++++++++++++++

def train_UNet_LSTM_model():

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    patch_size = PATCH_SIZE
    batch_size = BATCH_SIZE

    # =============================
    # 1. 加载固定数据划分
    # =============================

    train_coords = np.load(
        os.path.join(SPLIT_DIR, "train_coords_2021_seed42.npy")
    )

    val_coords = np.load(
        os.path.join(SPLIT_DIR, "val_coords_2021_seed42.npy")
    )

    test_coords = np.load(
        os.path.join(SPLIT_DIR, "test_coords_2021_seed42.npy")
    )

    print("\n📦 固定 2021 数据划分")
    print(f"🟢 训练: {len(train_coords)}")
    print(f"🟡 验证: {len(val_coords)}")
    print(f"🔵 测试: {len(test_coords)}")

    assert len(train_coords) == 5494
    assert len(val_coords) == 1832
    assert len(test_coords) == 1832

    # =============================
    # 2. 加载 2021 数据
    # =============================

    print("\n📦 加载 2021 数据")

    sar_dir = r"D:\WangRuozhen\RiceMapping\Data\SAR\Arkansas\2021x"
    label_path = r"D:\WangRuozhen\RiceMapping\Data\CDL\AR_CDL_2021_Clip.tif"

    sar_2021, _ = load_sar_data(
        sar_dir,
        year=2021,
        apply_filter=True
    )

    label_2021, _ = load_label_data(label_path)

    # =============================
    # 3. Generator
    # =============================

    train_gen = SARDataGenerator(
        sar_2021,
        label_2021,
        train_coords.copy(),
        batch_size=batch_size,
        shuffle=True,
        augment=True,
        patch_size=patch_size
    )

    val_gen = SARDataGenerator(
        sar_2021,
        label_2021,
        val_coords.copy(),
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        patch_size=patch_size
    )

    test_gen = SARDataGenerator(
        sar_2021,
        label_2021,
        test_coords.copy(),
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        patch_size=patch_size
    )

    print("\nGenerator batches:")
    print("Train:", len(train_gen))
    print("Validation:", len(val_gen))
    print("Test:", len(test_gen))

    # =============================
    # 4. 模型
    # =============================

    model = build_UNet_LSTM_model(
        input_shape=(patch_size, patch_size, sar_2021.shape[2])
    )

    quick_params_summary(model)

    def focal_loss_binary(alpha=0.75, gamma=2.0):
        def loss(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-4, 1.0 - 1e-4)
            bce = -(y_true * tf.math.log(y_pred) +
                    (1.0 - y_true) * tf.math.log(1.0 - y_pred))
            p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
            focal = alpha * tf.pow(1.0 - p_t, gamma) * bce
            return tf.reduce_mean(focal)

        return loss

    model.compile(
        optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
        loss=focal_loss_binary(alpha=0.75, gamma=2.0),
        metrics=['accuracy', precision, recall, fmeasure, kappa_metrics, OA, IoU]
    )

    # =============================
    # 5. 保存 best model
    # =============================

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model_filename = os.path.join(
        OUTPUT_DIR,
        f"best_UNet_LSTM_2021_fixed_seed42_{current_time}.h5"
    )

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),

        ModelCheckpoint(
            model_filename,
            monitor='val_loss',
            save_best_only=True
        ),

        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3
        ),

        TensorBoard(
            log_dir=os.path.join(
                LOG_DIR,
                f"UNet_LSTM_2021_fixed_{current_time}"
            )
        )
    ]

    # =============================
    # 6. 训练
    # =============================

    print("\n🚀 开始训练...")

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=60,
        callbacks=callbacks,
        verbose=1
    )

    # =============================
    # 7. 重新加载最佳模型
    # =============================

    print("\n📦 加载最佳模型：")
    print(model_filename)

    custom_objects = {
        'precision': precision,
        'recall': recall,
        'fmeasure': fmeasure,
        'kappa_metrics': kappa_metrics,
        'OA': OA,
        'IoU': IoU
    }

    best_model = tf.keras.models.load_model(
        model_filename,
        custom_objects=custom_objects,
        compile=False
    )

    best_model.compile(
        optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy', precision, recall, fmeasure, kappa_metrics, OA, IoU]
    )

    # =============================
    # 8. 普通测试
    # =============================

    print("\n🧪 固定测试集普通评估")

    results = best_model.evaluate(
        test_gen,
        verbose=1
    )

    for name, val in zip(
        ['Loss', 'Acc', 'Precision', 'Recall', 'F1', 'Kappa', 'OA', 'IoU'],
        results
    ):
        print(f"{name}: {val:.6f}")

    # =============================
    # 9. 论文最终指标
    # =============================

    print("\n🧪 测试集整体指标 + patch bootstrap")

    patch_metrics = evaluate_global_metrics_with_patch_bootstrap(
        best_model,
        test_gen,
        threshold=THRESHOLD,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED
    )

    # =============================
    # 10. 保存 Fig.11 数据
    # =============================

    model_basename = os.path.basename(model_filename)
    model_tag = os.path.splitext(model_basename)[0]

    confidence_metrics = save_test_predictions(
        best_model,
        test_gen,
        OUTPUT_DIR,
        model_tag=model_tag,
        threshold=THRESHOLD
    )

    # =============================
    # 11. 一致性检查
    # =============================

    print("\n" + "=" * 75)
    print("Table 与 Fig.11 数据一致性检查")
    print("=" * 75)

    print(f"Bootstrap OA : {patch_metrics['Accuracy']:.6f}")
    print(f"Confidence OA: {confidence_metrics['OA']:.6f}")

    print(f"Bootstrap F1 : {patch_metrics['F1']:.6f}")
    print(f"Confidence F1: {confidence_metrics['F1']:.6f}")

    oa_diff = abs(patch_metrics["Accuracy"] - confidence_metrics["OA"])
    f1_diff = abs(patch_metrics["F1"] - confidence_metrics["F1"])

    print("OA difference:", oa_diff)
    print("F1 difference:", f1_diff)

    if oa_diff < 1e-8 and f1_diff < 1e-8:
        print("✅ Table指标与Fig.11数据完全一致")
    else:
        print("⚠️ Table与Fig.11数据存在差异")

    print("\n✅ UNet+LSTM 2021 → 2021 实验完成")
    print("Best model:", model_filename)

    return best_model, model_filename, patch_metrics, confidence_metrics


# +++++++++++++++++++++++++++++++++++++++++++
#     已保存模型重新测试
#     不重新划分 test set
# +++++++++++++++++++++++++++++++++++++++++++

def evaluate_saved_model(model_path):

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    custom_objects = {
        'precision': precision,
        'recall': recall,
        'fmeasure': fmeasure,
        'kappa_metrics': kappa_metrics,
        'OA': OA,
        'IoU': IoU
    }

    print("\n📦 正在加载模型：")
    print(model_path)

    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy', precision, recall, fmeasure, kappa_metrics, OA, IoU]
    )

    sar_2021, _ = load_sar_data(
        r"D:\WangRuozhen\RiceMapping\Data\SAR\Arkansas\2021x",
        year=2021,
        apply_filter=True
    )

    label_2021, _ = load_label_data(
        r"D:\WangRuozhen\RiceMapping\Data\CDL\AR_CDL_2021_Clip.tif"
    )

    test_coords = np.load(
        os.path.join(
            SPLIT_DIR,
            "test_coords_2021_seed42.npy"
        )
    )

    test_gen = SARDataGenerator(
        sar_2021,
        label_2021,
        test_coords.copy(),
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False,
        patch_size=PATCH_SIZE
    )

    print("🔵 固定测试样本数:", len(test_coords))

    patch_metrics = evaluate_global_metrics_with_patch_bootstrap(
        model,
        test_gen,
        threshold=THRESHOLD,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED
    )

    model_tag = os.path.splitext(os.path.basename(model_path))[0]

    confidence_metrics = save_test_predictions(
        model,
        test_gen,
        OUTPUT_DIR,
        model_tag=model_tag,
        threshold=THRESHOLD
    )

    return patch_metrics, confidence_metrics


# +++++++++++++++++++++++++++++++++++++++++++
#     UNet+LSTM 全图预测
#     用于分类图、概率图和局部放大图
# +++++++++++++++++++++++++++++++++++++++++++

def predict_full_image(model_path, output_path, YEAR=2021, batch_size=10, patch=128, stride=64, threshold=0.5):

    start_time = time.time()

    custom_objects = {
        'precision': precision,
        'recall': recall,
        'fmeasure': fmeasure,
        'kappa_metrics': kappa_metrics,
        'OA': OA,
        'IoU': IoU
    }

    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False
    )

    # =============================
    # 加载整幅2021 SAR
    # 修改：apply_filter=False → True
    # 与训练、测试保持完全一致
    # =============================

    sar_dir = rf'D:\WangRuozhen\RiceMapping\Data\SAR\Arkansas\{YEAR}x'

    sar_data, profile = load_sar_data(
        sar_dir,
        year=YEAR,
        apply_filter=True
    )

    height, width, channels = sar_data.shape

    pred_full = np.zeros((height, width), dtype=np.float32)

    # 修改：uint8 → uint16，更安全
    count = np.zeros((height, width), dtype=np.uint16)

    margin = patch // 4

    ys = list(range(0, height - patch + 1, stride))
    xs = list(range(0, width - patch + 1, stride))

    if ys[-1] != height - patch:
        ys.append(height - patch)

    if xs[-1] != width - patch:
        xs.append(width - patch)

    # 模型预热
    dummy = tf.zeros((batch_size, patch, patch, channels))
    _ = model(dummy, training=False)

    patches = []
    coords = []

    # =============================
    # 滑窗预测
    # =============================

    for y in tqdm(ys, desc="Sliding Window"):

        for x in xs:

            X = sar_data[y:y+patch, x:x+patch, :]

            if X.shape[:2] != (patch, patch):
                continue

            patches.append(X)
            coords.append((y, x))

            if len(patches) == batch_size:

                patches_array = np.asarray(patches, dtype=np.float32)

                preds = model(
                    tf.convert_to_tensor(patches_array),
                    training=False
                ).numpy()

                for (yy, xx), p in zip(coords, preds):
                    # 内部patch只取中心区域，边界patch保留影像边缘
                    py0 = 0 if yy == 0 else margin
                    py1 = patch if yy == height - patch else patch - margin
                    px0 = 0 if xx == 0 else margin
                    px1 = patch if xx == width - patch else patch - margin

                    pred_full[
                    yy + py0:yy + py1,
                    xx + px0:xx + px1
                    ] += p[
                         py0:py1,
                         px0:px1,
                         0
                         ]

                    count[
                    yy + py0:yy + py1,
                    xx + px0:xx + px1
                    ] += 1

                patches = []
                coords = []

    # =============================
    # 处理最后不足一个batch的patch
    # =============================

    if len(patches) > 0:

        patches_array = np.asarray(patches, dtype=np.float32)

        preds = model(
            tf.convert_to_tensor(patches_array),
            training=False
        ).numpy()

        for (yy, xx), p in zip(coords, preds):
            # 内部patch只取中心区域，边界patch保留影像边缘
            py0 = 0 if yy == 0 else margin
            py1 = patch if yy == height - patch else patch - margin
            px0 = 0 if xx == 0 else margin
            px1 = patch if xx == width - patch else patch - margin

            pred_full[
            yy + py0:yy + py1,
            xx + px0:xx + px1
            ] += p[
                 py0:py1,
                 px0:px1,
                 0
                 ]

            count[
            yy + py0:yy + py1,
            xx + px0:xx + px1
            ] += 1

    # =============================
    # 重叠区域取平均概率
    # =============================

    pred_prob = np.divide(
        pred_full,
        count,
        out=np.zeros_like(pred_full),
        where=count > 0
    )

    # 修改：与测试阶段统一采用 >= 0.5
    pred_class = (pred_prob >= threshold).astype(np.uint8)

    # =============================
    # 概率图路径
    # =============================

    prob_path = output_path.replace(
        "rice_prediction",
        "rice_probability"
    )

    # =============================
    # 保存概率图
    # =============================

    prob_profile = profile.copy()
    prob_profile.update(
        dtype='float32',
        count=1,
        nodata=None
    )

    with rasterio.open(prob_path, 'w', **prob_profile) as dst:
        dst.write(pred_prob.astype(np.float32), 1)

    # =============================
    # 保存分类图
    # 0 = Non-rice
    # 1 = Rice
    #
    # 所以不能把0再设置成NoData
    # =============================

    class_profile = profile.copy()
    class_profile.update(
        dtype='uint8',
        count=1,
        nodata=None
    )

    with rasterio.open(output_path, 'w', **class_profile) as dst:
        dst.write(pred_class, 1)

    print("✅ 分类图:", output_path)
    print("✅ 概率图:", prob_path)
    print(f"⏱ 运行时间: {(time.time() - start_time) / 60:.2f} 分钟")


# +++++++++++++++++++++++++++++++++++++++++++
#     Main
# +++++++++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':

    physical_devices = tf.config.list_physical_devices('GPU')

    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # 正式训练 + 固定20%测试
    model, model_filename, patch_metrics, confidence_metrics = train_UNet_LSTM_model()

    # 全图预测
    model_tag = os.path.splitext(os.path.basename(model_filename))[0]
    out_tif = os.path.join(OUTPUT_DIR, f'rice_prediction_{model_tag}.tif')

    predict_full_image(
        model_filename,
        out_tif,
        YEAR=2021,
        batch_size=10,
        patch=128,
        stride=64,
        threshold=0.5
    )

    # ========================================================
    # 如果以后只测试已有模型：
    # 把上面的训练和预测注释掉，
    # 再把下面 False 改成 True
    # ========================================================

    if False:

        model_filename = r"F:\RiceMapping\Code\Code_21\UNet\best_UNet_LSTM_2021_fixed_seed42_20261008-101530.h5"

        patch_metrics, confidence_metrics = evaluate_saved_model(model_filename)

        model_tag = os.path.splitext(os.path.basename(model_filename))[0]
        out_tif = os.path.join(OUTPUT_DIR, f'rice_prediction_{model_tag}.tif')

        predict_full_image(
            model_filename,
            out_tif,
            YEAR=2021,
            batch_size=10,
            patch=128,
            stride=64,
            threshold=0.5
        )