from copy import deepcopy
from socket import *
import threading
from model_train import load_model
import struct
import numpy as np
import time

# 实时推理类
class Inference_Realtime:
    def __init__(self):
        self.sEMG_list = np.zeros((8, 32))
        self.counter = 0
        self.start_done = 0
        self.predictor = load_model("./MODEL/model.pkl")
        self.infer_flag = 0
        self.tcp_connect()

        self.sEMG_max = -np.inf
        self.sEMG_min = np.inf

    def _zero_crossing(self, x, thr):
        sign = np.sign(x)
        diff_sign = np.diff(sign, axis=-1) != 0
        amp_diff = np.abs(np.diff(x, axis=-1)) > thr
        return np.sum(diff_sign & amp_diff, axis=-1).astype(np.float64)

    def _slope_sign_change(self, x, thr):
        dx = np.diff(x, axis=-1)
        left = dx[:, :-1] if x.ndim == 2 else dx[:-1]
        right = dx[:, 1:] if x.ndim == 2 else dx[1:]
        sign_chg = (left * right) < 0
        amp_ok = (np.abs(left) > thr) & (np.abs(right) > thr)
        return np.sum(sign_chg & amp_ok, axis=-1).astype(np.float64)

    # 特征计算
    def extract_features(self, emg: np.ndarray, threshold: float = 1e-6):
        if emg.ndim == 1:
            emg = emg[np.newaxis, :]

        # MAV RMS WL ZCR SSC
        return np.concatenate([np.mean(np.abs(emg), axis=-1),
                               np.sqrt(np.mean(emg ** 2, axis=-1)),
                               np.sum(np.abs(np.diff(emg, axis=-1)), axis=-1),
                               self._zero_crossing(emg, threshold),
                               self._slope_sign_change(emg, threshold),
                               ], axis=-1)

    # TCP连接
    def tcp_connect(self):
        self.tcp_server = socket(AF_INET, SOCK_STREAM)
        self.tcp_server.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        tcp_address = ('0.0.0.0', 9527)
        self.tcp_server.bind(tcp_address)
        self.tcp_server.listen(50)

        print("socket初始化成功")

        self.client_scoket, self.client_address = self.tcp_server.accept()
        print("socket连接成功:", self.client_address)

    # TCP数据接收
    def tcp_receive(self):
        while True:
            start = time.perf_counter()
            sEMG_ = [[] for i in range(8)]
            data = self.client_scoket.recv(29)
            idx = 0
            if data[idx] == 0xaa and data[idx + 1] == 0xaa and data[idx + 2] == 0xf1:
                for j in range(8):
                    b = data[idx + 4 + 3 * j:idx + 7 + 3 * j] + bytearray(1)
                    r = struct.unpack('>i', b)[0] / 256
                    sEMG_[j].append(r)

            self.sEMG_list[:, self.counter] = np.array(sEMG_).squeeze(1)
            self.counter += 1
            if self.start_done == 0:
                if self.counter == 32:
                    self.counter = 0
                    self.start_done = 1
            else:
                if self.counter == 32:
                    self.counter = 0
                if self.counter % 16 == 0:
                    self.infer_flag = 1

            end = time.perf_counter()
            # print(f"Data Received Time: {end - start:.6f} s")

    # 实时推理
    def inference(self):

        TH = 750  # TODO 根据经验调整激活阈值
        while True:
            if self.start_done == 1:

                sEMG = deepcopy(self.sEMG_list)
                sEMG = sEMG * 0.0397

                # 粗略计算激活值（未进行归一化）
                f0 = self.extract_features(sEMG)
                act = np.mean(f0)
                print(act)

                # 动态归一化
                if np.max(sEMG) > self.sEMG_max:
                    self.sEMG_max = np.max(sEMG)
                if np.min(sEMG) < self.sEMG_min:
                    self.sEMG_min = np.min(sEMG)

                sEMG = (sEMG - np.mean(sEMG, axis=1)[:, np.newaxis]) / (self.sEMG_max - self.sEMG_min)

                # 归一化后特征计算
                f = self.extract_features(sEMG)
                if act > TH:
                    result = self.predictor.predict_one(f)
                    self.client_scoket.send(str(result['label']).encode('utf-8'))
                else:
                    self.client_scoket.send("0".encode('utf-8'))
                # print(result)
                self.start_done = 0
                end = time.perf_counter()

                # 打印推理时间
                # print(f"Inference Time: {end - start:.6f} s")

    def inference_run(self):
        # 多线程运行
        threading.Thread(target=self.tcp_receive).start()
        threading.Thread(target=self.inference).start()

if __name__=='__main__':
    infer = Inference_Realtime()
    infer.inference_run()