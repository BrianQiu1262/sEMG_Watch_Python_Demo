import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import pickle as pkl

# 特征计算，五种sEMG典型时域特征MAV RMS WL ZCR SSC
def extract_features(emg: np.ndarray, threshold: float = 1e-6):
    """
    emg : shape (8, N)  —— 8 条 sEMG，每行一通道
    返回 : dict，每个值是长度 8 的数组
    """
    if emg.ndim == 1:
        emg = emg[np.newaxis, :]

    # MAV RMS WL ZCR SSC
    return np.concatenate([np.mean(np.abs(emg), axis=-1),
                           np.sqrt(np.mean(emg ** 2, axis=-1)),
                           np.sum(np.abs(np.diff(emg, axis=-1)), axis=-1),
                           _zero_crossing(emg, threshold),
                           _slope_sign_change(emg, threshold),
                           ], axis=-1)


def _zero_crossing(x, thr):
    sign = np.sign(x)
    diff_sign = np.diff(sign, axis=-1) != 0
    amp_diff = np.abs(np.diff(x, axis=-1)) > thr
    return np.sum(diff_sign & amp_diff, axis=-1).astype(np.float64)


def _slope_sign_change( x, thr):
    dx = np.diff(x, axis=-1)
    left = dx[:, :-1] if x.ndim == 2 else dx[:-1]
    right = dx[:, 1:] if x.ndim == 2 else dx[1:]
    sign_chg = (left * right) < 0
    amp_ok = (np.abs(left) > thr) & (np.abs(right) > thr)
    return np.sum(sign_chg & amp_ok, axis=-1).astype(np.float64)

# 读取CSV文件
file_path = './RAW/data_1_6.csv'   #TODO 读取C#上位机保留的原始数据
df = pd.read_csv(file_path)

# 采样频率
fs = 250

sEMG = [[] for i in range(8)]

semg_cols = [col for col in df.columns if col.startswith('sEMG')]
for idx, col in enumerate(semg_cols):
    sEMG[idx] = df[col]

# 绘图 - 8通道sEMG数据
fig, axes = plt.subplots(8, 1, figsize=(12, 15), sharex=True)
fig.suptitle('sEMG', fontsize=18, fontweight='bold', y=0.98)

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
          '#9b59b6', '#1abc9c', '#e67e22', '#34495e']

for i, col in enumerate(semg_cols):
    ax = axes[i]
    ax.plot(df['Time'], df[col], color=colors[i], linewidth=0.6)
    ax.set_ylabel(col, fontsize=12, fontweight='bold')
    ax.set_ylim(-2000, 2000)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[-1].set_xlabel('Time (s)', fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()

# 数据归一化
sEMG = np.array(sEMG)  * 0.0397
sEMG0 = copy.deepcopy(sEMG)
sEMG = (sEMG - np.mean(sEMG, axis=1)[:, np.newaxis]) / (np.max(sEMG) - np.min(sEMG))

# 根据绘图中的8条曲线，确定激活起始点START与终点END，同时确定滑窗窗宽
START = 0.9 * fs #TODO 有效数据起始点
END = 1.5 * fs #TODO 有效数据终点
WIN = 32 #TODO 滑窗窗宽

f = []
act = []

# 特征计算以及激活区域选取
for n in range(int(sEMG.shape[1]/WIN)):
    start_point = int(n * WIN)
    end_point = int((n + 1) * WIN)
    f.append(extract_features(sEMG[:, start_point:end_point]))
    if start_point > START and end_point < END:
        act.append(1)
    else:
        act.append(0)

features = np.array(f).transpose((1, 0))

# 数据储存
save_file = './DATA/1/d6.pkl' #TODO 将特征、激活区域、原始信号保存到./DATA下，并按照手势类别存入不同文件夹以待训练
data_dict = {'data': features, 'act': act, 'raw_data': sEMG0}
with open(save_file, 'wb') as f:
    pkl.dump(data_dict, f)
