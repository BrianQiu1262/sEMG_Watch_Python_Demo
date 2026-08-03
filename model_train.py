import argparse
import os
import pickle
import sys
import time

import numpy as np


DEFAULT_MODEL_NAME = "./MODEL/model.pkl" # TODO 模型存放地址
DEFAULT_CLASS_NAMES = (1, 2, 3, 4) # TODO 手势类别，应随着DATA下的文件名一起改

# 模型训练、推理、保存类
class FastLinear4Class(object):
    """
    线性四分类器。

    训练时使用标准化 + 岭最小二乘；
    保存前将标准化合并到权重中，因此实时推理无需再标准化。
    """

    def __init__(self):
        self.classes = None
        self.coefficients = None
        self.intercept = None
        self.feature_count = None

    # 模型训练
    def fit(self, x, y, ridge=0.1):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).reshape(-1)

        if x.ndim != 2:
            raise ValueError("训练特征必须是二维数组")

        if len(x) != len(y):
            raise ValueError("训练特征与标签数量不一致")

        self.classes = np.unique(y)

        if len(self.classes) != 4:
            raise ValueError(
                "训练数据必须包含4个类别，当前类别为 {}".format(
                    self.classes.tolist()
                )
            )

        self.feature_count = int(x.shape[1])

        feature_mean = x.mean(axis=0)
        feature_std = x.std(axis=0)
        feature_std[feature_std < 1e-12] = 1.0

        x_normalized = (
            x - feature_mean
        ) / feature_std

        sample_count = x_normalized.shape[0]
        class_count = len(self.classes)

        # One-hot 目标矩阵
        target = np.zeros(
            (sample_count, class_count),
            dtype=np.float64
        )

        for class_index, class_id in enumerate(
            self.classes
        ):
            target[
                y == class_id,
                class_index
            ] = 1.0

        # 添加偏置列
        design_matrix = np.empty(
            (
                sample_count,
                self.feature_count + 1
            ),
            dtype=np.float64
        )
        design_matrix[:, :-1] = x_normalized
        design_matrix[:, -1] = 1.0

        regularization = np.eye(
            self.feature_count + 1,
            dtype=np.float64
        ) * float(ridge)

        # 不对偏置项做正则化
        regularization[-1, -1] = 0.0

        lhs = (
            np.dot(
                design_matrix.T,
                design_matrix
            )
            + regularization
        )
        rhs = np.dot(
            design_matrix.T,
            target
        )

        try:
            weights = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            weights = np.dot(
                np.linalg.pinv(lhs),
                rhs
            )

        normalized_coefficients = weights[:-1, :]
        normalized_intercept = weights[-1, :]

        # 将标准化折叠进线性权重：
        # ((x-mean)/std) @ W + b
        # = x @ (W/std) + [b - (mean/std)@W]
        raw_coefficients = (
            normalized_coefficients
            / feature_std[:, np.newaxis]
        )

        raw_intercept = (
            normalized_intercept
            - np.dot(
                feature_mean / feature_std,
                normalized_coefficients
            )
        )

        # 实时推理采用 float32，速度更快且内存更小
        self.coefficients = np.ascontiguousarray(
            raw_coefficients,
            dtype=np.float32
        )
        self.intercept = np.asarray(
            raw_intercept,
            dtype=np.float32
        )

        return self

    def decision_function(self, x):
        if (
            self.classes is None
            or self.coefficients is None
            or self.intercept is None
        ):
            raise RuntimeError("模型尚未训练或加载")

        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.ndim != 2:
            raise ValueError("输入必须是一维或二维特征数组")

        if x.shape[1] != self.feature_count:
            raise ValueError(
                "输入特征数为 {}，模型要求 {}".format(
                    x.shape[1],
                    self.feature_count
                )
            )

        return (
            np.dot(x, self.coefficients)
            + self.intercept
        )

    # 批量预测
    def predict(self, x):
        scores = self.decision_function(x)
        class_indices = np.argmax(scores, axis=1)
        return self.classes[class_indices]

    # 单次预测
    def predict_one(self, feature_window):
        feature_window = np.asarray(
            feature_window,
            dtype=np.float32
        ).reshape(-1)

        if len(feature_window) != self.feature_count:
            raise ValueError(
                "输入窗口包含 {} 个特征，模型要求 {}".format(
                    len(feature_window),
                    self.feature_count
                )
            )

        start_time = time.perf_counter()

        scores = (
            np.dot(
                feature_window,
                self.coefficients
            )
            + self.intercept
        )

        class_index = int(np.argmax(scores))
        label = int(self.classes[class_index])

        elapsed_us = (
            time.perf_counter() - start_time
        ) * 1000000.0

        score_dict = {}
        for index, class_id in enumerate(
            self.classes
        ):
            score_dict[int(class_id)] = float(
                scores[index]
            )

        return {
            "label": label,
            "scores": score_dict,
            "elapsed_us": elapsed_us
        }


def natural_sort_key(filename):
    """使 d2.pkl 排在 d10.pkl 前面。"""
    digits = ""
    for character in filename:
        if character.isdigit():
            digits += character

    if digits:
        return int(digits)

    return filename


def orient_windows(data, act, pkl_path):
    """
    将 data 统一转换为：
        (窗口数量, 特征数量)

    自动兼容：
        (15, 特征数)
        (特征数, 15)
    """
    data = np.asarray(data, dtype=np.float32)
    act = np.asarray(act).reshape(-1)

    if data.ndim != 2:
        raise ValueError(
            "{} 中 data 必须是二维数组，当前形状为 {}".format(
                pkl_path,
                data.shape
            )
        )

    if data.shape[0] == len(act):
        windows = data
    elif data.shape[1] == len(act):
        windows = data.T
    else:
        raise ValueError(
            "{} 中 data 形状 {} 无法与 act 长度 {} 对应".format(
                pkl_path,
                data.shape,
                len(act)
            )
        )

    return np.ascontiguousarray(
        windows,
        dtype=np.float32
    ), act

# 读取文件，提取数据
def load_one_pkl_valid_windows(pkl_path):
    """读取一个 pkl，只返回 act==1 的有效小窗。"""
    with open(pkl_path, "rb") as file_obj:
        sample_dict = pickle.load(file_obj)

    if not isinstance(sample_dict, dict):
        raise TypeError(
            "{} 的内容不是 dict".format(pkl_path)
        )

    if (
        "data" not in sample_dict
        or "act" not in sample_dict
    ):
        raise KeyError(
            "{} 必须包含 data 和 act".format(
                pkl_path
            )
        )

    windows, act = orient_windows(
        sample_dict["data"],
        sample_dict["act"],
        pkl_path
    )

    valid_mask = act == 1
    valid_windows = windows[valid_mask]

    return (
        valid_windows,
        valid_mask,
        windows.shape[0],
        windows.shape[1]
    )

# 构建数据集
def load_dataset(data_dir):
    """
    遍历 DATA/1～DATA/4，提取全部有效小窗。
    """
    feature_blocks = []
    label_blocks = []
    records = []
    feature_count = None

    class_sample_counts = {}
    class_file_counts = {}

    for class_id in DEFAULT_CLASS_NAMES:
        class_dir = os.path.join(
            data_dir,
            str(class_id)
        )

        if not os.path.isdir(class_dir):
            raise IOError(
                "找不到类别目录：{}".format(class_dir)
            )

        pkl_files = [
            filename
            for filename in os.listdir(class_dir)
            if filename.lower().endswith(".pkl")
        ]
        pkl_files.sort(key=natural_sort_key)

        if not pkl_files:
            raise IOError(
                "{} 中没有找到 pkl 文件".format(
                    class_dir
                )
            )

        class_file_counts[class_id] = len(pkl_files)
        class_sample_counts[class_id] = 0

        for filename in pkl_files:
            pkl_path = os.path.join(
                class_dir,
                filename
            )

            (
                valid_windows,
                valid_mask,
                total_window_count,
                current_feature_count
            ) = load_one_pkl_valid_windows(pkl_path)

            if feature_count is None:
                feature_count = current_feature_count
            elif current_feature_count != feature_count:
                raise ValueError(
                    "{} 的特征数为 {}，此前特征数为 {}".format(
                        pkl_path,
                        current_feature_count,
                        feature_count
                    )
                )

            valid_count = len(valid_windows)

            if valid_count > 0:
                feature_blocks.append(valid_windows)
                label_blocks.append(
                    np.full(
                        valid_count,
                        class_id,
                        dtype=np.int64
                    )
                )

            class_sample_counts[class_id] += valid_count

            records.append({
                "class_id": class_id,
                "filename": filename,
                "total_windows": total_window_count,
                "valid_windows": valid_count,
                "valid_indices": np.flatnonzero(
                    valid_mask
                ).tolist()
            })

    if not feature_blocks:
        raise ValueError(
            "没有找到任何 act==1 的有效窗口"
        )

    x_all = np.ascontiguousarray(
        np.vstack(feature_blocks),
        dtype=np.float32
    )
    y_all = np.concatenate(label_blocks)

    return (
        x_all,
        y_all,
        records,
        class_sample_counts,
        class_file_counts
    )

# 混淆矩阵计算
def confusion_matrix_4class(y_true, y_pred):
    matrix = np.zeros((4, 4), dtype=np.int64)

    for row, true_label in enumerate(
        DEFAULT_CLASS_NAMES
    ):
        for col, predicted_label in enumerate(
            DEFAULT_CLASS_NAMES
        ):
            matrix[row, col] = np.sum(
                (y_true == true_label)
                & (y_pred == predicted_label)
            )

    return matrix


def print_confusion_matrix(matrix):
    print("")
    print("混淆矩阵（行=真实类别，列=预测类别）")
    print("          预测1  预测2  预测3  预测4")

    for index, class_id in enumerate(
        DEFAULT_CLASS_NAMES
    ):
        print(
            "真实{}    {:5d}  {:5d}  {:5d}  {:5d}".format(
                class_id,
                int(matrix[index, 0]),
                int(matrix[index, 1]),
                int(matrix[index, 2]),
                int(matrix[index, 3])
            )
        )

# 模型保存
def save_model(model, model_path):
    model_dict = {
        "classes": model.classes,
        "coefficients": model.coefficients,
        "intercept": model.intercept,
        "feature_count": model.feature_count,
        "model_type": "linear_ridge_4class"
    }

    with open(model_path, "wb") as file_obj:
        pickle.dump(
            model_dict,
            file_obj,
            protocol=2
        )

# 模型加载
def load_model(model_path):
    with open(model_path, "rb") as file_obj:
        model_dict = pickle.load(file_obj)

    required_keys = (
        "classes",
        "coefficients",
        "intercept",
        "feature_count"
    )

    for key in required_keys:
        if key not in model_dict:
            raise KeyError(
                "模型文件缺少参数：{}".format(key)
            )

    model = FastLinear4Class()
    model.classes = np.asarray(
        model_dict["classes"]
    )
    model.coefficients = np.ascontiguousarray(
        model_dict["coefficients"],
        dtype=np.float32
    )
    model.intercept = np.asarray(
        model_dict["intercept"],
        dtype=np.float32
    )
    model.feature_count = int(
        model_dict["feature_count"]
    )

    return model


def benchmark_model(model, x, repeat=100000):
    repeat_count = int(
        np.ceil(
            float(repeat) / len(x)
        )
    )

    benchmark_x = np.tile(
        x,
        (repeat_count, 1)
    )[:repeat]

    model.predict(
        benchmark_x[:min(1000, repeat)]
    )

    start_time = time.perf_counter()
    model.predict(benchmark_x)
    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    return (
        elapsed_seconds * 1000.0,
        elapsed_seconds
        / repeat
        * 1000000.0
    )

# 训练和评估
def train_and_evaluate(
    data_dir,
    model_path,
    ridge
):
    (
        x_all,
        y_all,
        records,
        class_sample_counts,
        class_file_counts
    ) = load_dataset(data_dir)

    print("数据读取完成")
    print("-" * 60)

    for record in records:
        print(
            "动作{} / {}：总窗数={}，有效窗数={}，有效索引={}".format(
                record["class_id"],
                record["filename"],
                record["total_windows"],
                record["valid_windows"],
                record["valid_indices"]
            )
        )

    print("")
    print("训练数据汇总")
    print("-" * 60)

    for class_id in DEFAULT_CLASS_NAMES:
        print(
            "动作{}：{} 个文件，{} 个有效窗口".format(
                class_id,
                class_file_counts[class_id],
                class_sample_counts[class_id]
            )
        )

    print(
        "总有效样本数：{}".format(
            len(x_all)
        )
    )
    print(
        "每个窗口特征数：{}".format(
            x_all.shape[1]
        )
    )

    model = FastLinear4Class()
    model.fit(
        x_all,
        y_all,
        ridge=ridge
    )

    '''
    注意！！！在此使用的测试集和训练集相同，仅为走通模型训练流程作展示，
    若严谨地分析模型泛化性需要从原始数据中划分出不同的训练、测试集
    '''
    y_pred = model.predict(x_all)

    correct_count = int(
        np.sum(y_pred == y_all)
    )
    accuracy = float(
        np.mean(y_pred == y_all)
    )

    matrix = confusion_matrix_4class(
        y_all,
        y_pred
    )

    print("")
    print("同一批有效数据训练与测试结果")
    print("-" * 60)
    print(
        "识别正确：{}/{}".format(
            correct_count,
            len(y_all)
        )
    )
    print(
        "识别准确率：{:.2f}%".format(
            accuracy * 100.0
        )
    )

    print_confusion_matrix(matrix)

    save_model(model, model_path)

    print("")
    print(
        "模型已保存：{}".format(
            model_path
        )
    )

    total_ms, us_per_window = benchmark_model(
        model,
        x_all
    )

    print("")
    print("推理速度参考")
    print("-" * 60)
    print(
        "批量推理100000个窗口耗时：{:.3f} ms".format(
            total_ms
        )
    )
    print(
        "平均每个窗口：{:.3f}微秒".format(
            us_per_window
        )
    )

    return model


def print_prediction(result, sequence_number):
    score_text = []

    for class_id in DEFAULT_CLASS_NAMES:
        score_text.append(
            "动作{}={:.6f}".format(
                class_id,
                result["scores"][class_id]
            )
        )

    print(
        "[窗口 {:04d}] 预测=动作{} | {} | 推理={:.3f}微秒".format(
            sequence_number,
            result["label"],
            ", ".join(score_text),
            result["elapsed_us"]
        )
    )

# 实时推理demo测试
def run_demo(model, data_dir):
    x_all, y_all, _, _, _ = load_dataset(
        data_dir
    )

    correct_count = 0

    print("")
    print("开始逐窗口模拟实时推理")
    print("-" * 60)

    for index in range(len(x_all)):
        result = model.predict_one(
            x_all[index]
        )

        if result["label"] == int(
            y_all[index]
        ):
            correct_count += 1

        print_prediction(
            result,
            index + 1
        )

    accuracy = (
        float(correct_count)
        / len(x_all)
        * 100.0
    )

    print("")
    print(
        "回放完成：{}/{} 正确，准确率 {:.2f}%".format(
            correct_count,
            len(x_all),
            accuracy
        )
    )

# 命令行推理测试
def parse_feature_line(line):
    line = line.strip()

    if not line:
        return None

    line = line.replace(",", " ")
    parts = line.split()

    try:
        values = [
            float(value)
            for value in parts
        ]
    except ValueError:
        raise ValueError(
            "输入中包含无法转换为数字的内容"
        )

    return np.asarray(
        values,
        dtype=np.float32
    )


def run_stdin(model):
    print("")
    print("实时推理模式")
    print(
        "每行输入一个 {} 维特征窗口，"
        "使用逗号或空格分隔；输入 q 退出。".format(
            model.feature_count
        )
    )

    sequence_number = 0

    while True:
        try:
            line = input("sEMG> ")
        except EOFError:
            break
        except KeyboardInterrupt:
            print("")
            break

        if line.strip().lower() in (
            "q",
            "quit",
            "exit"
        ):
            break

        try:
            feature_window = parse_feature_line(
                line
            )

            if feature_window is None:
                continue

            sequence_number += 1

            result = model.predict_one(
                feature_window
            )
            print_prediction(
                result,
                sequence_number
            )

        except Exception as error:
            print(
                "输入或推理错误：{}".format(error)
            )

# pkl文件推理测试
def predict_pkl(model, pkl_path):
    (
        valid_windows,
        valid_mask,
        total_window_count,
        feature_count
    ) = load_one_pkl_valid_windows(pkl_path)

    if feature_count != model.feature_count:
        raise ValueError(
            "pkl中特征数为{}，模型要求{}".format(
                feature_count,
                model.feature_count
            )
        )

    print(
        "{}：总窗口数={}，有效窗口数={}".format(
            pkl_path,
            total_window_count,
            len(valid_windows)
        )
    )
    print(
        "有效窗口索引：{}".format(
            np.flatnonzero(valid_mask).tolist()
        )
    )

    for index in range(len(valid_windows)):
        result = model.predict_one(
            valid_windows[index]
        )
        print_prediction(
            result,
            index + 1
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "使用dict['data']与dict['act']"
            "训练四分类sEMG实时模型"
        )
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="重新训练模型并统计同集测试准确率"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="逐个回放DATA目录中的全部有效窗口"
    )

    parser.add_argument(
        "--stdin",
        action="store_true",
        help="从命令行持续输入单个特征窗口"
    )

    parser.add_argument(
        "--predict-pkl",
        default=None,
        help="对指定pkl中的全部有效窗口进行推理"
    )

    parser.add_argument(
        "--data-dir",
        default=None,
        help="DATA目录路径，默认读取代码同级的DATA"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="模型文件路径"
    )

    parser.add_argument(
        "--ridge",
        type=float,
        default=0.1,
        help="岭正则化系数，默认0.1"
    )

    return parser

# 主函数
def main():
    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    parser = build_parser()
    args = parser.parse_args()

    if args.data_dir is None:
        data_dir = os.path.join(
            base_dir,
            "DATA"
        )
    else:
        data_dir = os.path.abspath(
            args.data_dir
        )

    model_path = args.model

    if not os.path.isabs(model_path):
        model_path = os.path.join(
            base_dir,
            model_path
        )

    try:
        if args.train or not os.path.exists(
            model_path
        ):
            model = train_and_evaluate(
                data_dir,
                model_path,
                args.ridge
            )
        else:
            model = load_model(model_path)
            print(
                "已加载模型：{}".format(
                    model_path
                )
            )

        if args.demo:
            run_demo(model, data_dir)
        elif args.predict_pkl:
            predict_pkl(
                model,
                os.path.abspath(
                    args.predict_pkl
                )
            )
        elif args.stdin:
            run_stdin(model)
        elif not args.train:
            print("")
            print(
                "未指定推理模式。可使用 "
                "--demo、--stdin 或 --predict-pkl。"
            )

    except Exception as error:
        print(
            "程序错误：{}".format(error)
        )
        sys.exit(1)


if __name__ == "__main__":
    main()