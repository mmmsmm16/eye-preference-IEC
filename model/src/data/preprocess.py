import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from typing import List, Tuple, Dict, Any, Optional, Callable
import os


def remove_last_half_second(df: pd.DataFrame) -> pd.DataFrame:
    """
    データフレームの最後の0.5秒間のデータを削除する

    :param df: 入力のDataFrame
    :return: 最後の1.0秒間のデータが削除されたDataFrame
    """
    last_timestamp = df['timestamp'].max()
    cutoff_timestamp = last_timestamp - 1 * 10**6
    # cutoff_timestampよりも前のデータのみを残す
    return df[df['timestamp'] <= cutoff_timestamp]

def remove_last_ten_percent(df: pd.DataFrame) -> pd.DataFrame:
    """
    データフレームの最後の10%のデータを削除する

    :param df: 入力のDataFrame
    :return: 最後の10%のデータが削除されたDataFrame
    """
    last_timestamp = df['timestamp'].max()
    cutoff_timestamp = last_timestamp - 0.1 * (last_timestamp - df['timestamp'].min())
    # cutoff_timestampよりも前のデータのみを残す
    return df[df['timestamp'] <= cutoff_timestamp]

def trim_select_button_fixations(
    df: pd.DataFrame,
    button_regions: List[Dict[str, float]] = [
        # 各画像の右側にあるSELECTボタンの領域を定義
        {'x_min': 0.163, 'x_max': 0.225, 'y_min': 0.204, 'y_max': 0.292},  # 1番目の画像のボタン
        {'x_min': 0.7758, 'x_max': 0.837, 'y_min': 0.204, 'y_max': 0.292},  # 2番目の画像のボタン
        {'x_min': 0.163, 'x_max': 0.225, 'y_min': 0.708, 'y_max': 0.796},  # 3番目の画像のボタン
        {'x_min': 0.775, 'x_max': 0.837, 'y_min': 0.708, 'y_max': 0.796}   # 4番目の画像のボタン
    ]
) -> pd.DataFrame:
    """
    最後の1秒のデータを確認し、ボタン上に視点がある場合それ以降を削除する。
    全体が削除される場合は、さらにさかのぼってボタン上の視線の最初の時点を探す。
    
    :param df: 視線データのDataFrame
    :param button_regions: 拡大されたSELECTボタンの領域を定義する辞書のリスト
    :return: トリミングされたDataFrame
    """
    try:
        if df.empty:
            return df

        def is_in_button_region(x: float, y: float) -> bool:
            """座標がボタン領域内にあるかチェック"""
            return any(
                region['x_min'] <= x <= region['x_max'] and 
                region['y_min'] <= y <= region['y_max']
                for region in button_regions
            )

        # 最後の1秒のデータを抽出
        last_second_start = df['timestamp'].max() - 1_000_000  # 1秒 = 1,000,000 マイクロ秒
        last_second_data = df[df['timestamp'] >= last_second_start]
        
        if last_second_data.empty:
            return df
        
        # # デバッグ情報
        # print(f"\nAnalyzing last second of data:")
        # print(f"Total samples: {len(df)}")
        # print(f"Samples in last second: {len(last_second_data)}")
        
        # 最後の1秒のデータを時系列順に処理
        trim_timestamp = None
        first_button_gaze_in_last_second = None
        
        for _, row in last_second_data.iterrows():
            # 左右の視点の平均を計算
            x = (row['left_x'] + row['right_x']) / 2
            y = (row['left_y'] + row['right_y']) / 2
            
            # ボタン領域内にあるかチェック
            if is_in_button_region(x, y):
                if first_button_gaze_in_last_second is None:
                    first_button_gaze_in_last_second = row['timestamp']
                trim_timestamp = first_button_gaze_in_last_second
                break
        
        # もし最後の1秒で見つかった時点が最後の1秒の開始時点と同じ（≒全削除）なら
        # さらにさかのぼって最初のボタン視点を探す
        if trim_timestamp is not None and abs(trim_timestamp - last_second_start) < 1000:  # 1000マイクロ秒の誤差を許容
            # print("\nDetected potential full second deletion, searching earlier data...")
            
            # 最後の1秒より前のデータを時系列逆順に処理
            earlier_data = df[df['timestamp'] < last_second_start].sort_values('timestamp', ascending=False)
            
            # ボタンを見ていない最初の時点を探す（そこから次の時点がボタンを見始めた時点）
            last_non_button_timestamp = None
            for _, row in earlier_data.iterrows():
                x = (row['left_x'] + row['right_x']) / 2
                y = (row['left_y'] + row['right_y']) / 2
                
                if not is_in_button_region(x, y):
                    last_non_button_timestamp = row['timestamp']
                    break
            
            if last_non_button_timestamp is not None:
                # ボタンを見ていない最後の時点の次のデータポイントを取得
                first_button_gaze = df[df['timestamp'] > last_non_button_timestamp].iloc[0]
                trim_timestamp = first_button_gaze['timestamp']
                # print(f"Found earlier button gaze at: {(trim_timestamp - df['timestamp'].min()) / 1e6:.3f}s")
        
        # ボタン上の視点が見つかった場合、その時点でトリミング
        if trim_timestamp is not None:
            df_trimmed = df[df['timestamp'] < trim_timestamp]
            
            # # トリミング統計の出力
            # original_duration = (df['timestamp'].max() - df['timestamp'].min()) / 1e6
            # trimmed_duration = (df_trimmed['timestamp'].max() - df_trimmed['timestamp'].min()) / 1e6
            # removed_duration = original_duration - trimmed_duration
            
            # print(f"\nTrimming Statistics:")
            # print(f"Original duration: {original_duration:.2f}s")
            # print(f"Trimmed duration: {trimmed_duration:.2f}s")
            # print(f"Removed duration: {removed_duration:.2f}s")
            # print(f"Removed percentage: {(removed_duration/original_duration)*100:.2f}%")
            
            # # ボタン視点の追加統計
            # button_gazes_before = sum(1 for _, row in df.iterrows() 
            #     if is_in_button_region((row['left_x'] + row['right_x']) / 2, 
            #                          (row['left_y'] + row['right_y']) / 2))
            # button_gazes_after = sum(1 for _, row in df_trimmed.iterrows() 
            #     if is_in_button_region((row['left_x'] + row['right_x']) / 2, 
            #                          (row['left_y'] + row['right_y']) / 2))
            
            # print(f"\nButton Gaze Statistics:")
            # print(f"Original button gazes: {button_gazes_before}")
            # print(f"Remaining button gazes: {button_gazes_after}")
            # print(f"Removed button gazes: {button_gazes_before - button_gazes_after}")
            
            return df_trimmed
        
        # print("\nNo button gazes found in last second")
        return df

    except Exception as e:
        print(f"Error in trim_last_second_button: {str(e)}")
        return df

def trim_based_on_fixations(
    df: pd.DataFrame,
    button_regions: List[Dict[str, float]] = [
        {'x_min': 0, 'x_max': 0.225, 'y_min': 0, 'y_max': 1},
        {'x_min': 0.775, 'x_max': 1, 'y_min': 0, 'y_max': 1}
        # {'x_min': 0.163, 'x_max': 0.225, 'y_min': 0.204, 'y_max': 0.292},
        # {'x_min': 0.7758, 'x_max': 0.837, 'y_min': 0.204, 'y_max': 0.292},
        # {'x_min': 0.163, 'x_max': 0.225, 'y_min': 0.708, 'y_max': 0.796},
        # {'x_min': 0.775, 'x_max': 0.837, 'y_min': 0.708, 'y_max': 0.796}
    ],
    collect_stats: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    注視点に基づいてデータをトリミングし、統計情報を収集する
    
    :param df: 視線データのDataFrame
    :param button_regions: ボタン領域の定義
    :param collect_stats: 統計情報を収集するかどうか
    :return: (トリミングされたDataFrame, 統計情報の辞書)
    """
    stats = {
        'original_duration': 0,
        'trimmed_duration': 0,
        'total_fixations': 0,
        'button_fixations': 0,
        'trimming_type': 'none'  # トリミングの種類を記録
    }
    
    try:
        if df.empty:
            return df, stats

        def is_in_button_region(x: float, y: float) -> bool:
            return any(
                region['x_min'] <= x <= region['x_max'] and 
                region['y_min'] <= y <= region['y_max']
                for region in button_regions
            )

        # 統計情報の収集開始
        if collect_stats:
            stats['original_duration'] = (df['timestamp'].max() - df['timestamp'].min()) / 1e6

        # 全ての注視点を抽出
        fixations = extract_fixations(df)
        if not fixations:
            return df, stats

        stats['total_fixations'] = len(fixations)

        # ボタン領域上の注視点を検出
        button_fixations = []
        for i, fixation in enumerate(fixations):
            if is_in_button_region(fixation['x'], fixation['y']):
                button_fixations.append(i)
        
        stats['button_fixations'] = len(button_fixations)

        # トリミングポイントの決定
        if button_fixations:
            if len(button_fixations) > 1:
                # 最後のボタン注視点のインデックス
                last_button_idx = button_fixations[-1]
                prev_button_idx = button_fixations[-2]
                
                # 最後のボタン注視点とその一つ前のボタン注視点が連続しているか確認
                if last_button_idx == prev_button_idx + 1:
                    # 連続している場合、その一つ前の注視点までのデータを探す
                    if prev_button_idx > 0:
                        # prev_button_idxより前の注視点を取得
                        prev_fixation = fixations[prev_button_idx - 1]
                        trim_fixation = prev_fixation
                        stats['trimming_type'] = 'consecutive_end_button_fixations'
                    else:
                        return df, stats
                else:
                    # 連続していない場合、最後のボタン注視点の一つ前の注視点でトリミング
                    if last_button_idx > 0:
                        trim_fixation = fixations[last_button_idx - 1]
                        stats['trimming_type'] = 'multiple_button_fixations'
                    else:
                        return df, stats
            else:
                button_fixation_idx = button_fixations[0]
                if button_fixation_idx > 0:
                    trim_fixation = fixations[button_fixation_idx - 1]
                    stats['trimming_type'] = 'single_button_fixation'
                else:
                    return df, stats
        else:
            trim_fixation = fixations[-1]
            stats['trimming_type'] = 'no_button_fixation'

        # データのトリミング
        df_trimmed = df[df['timestamp'] <= trim_fixation['end_time']]

        # 最終的な統計情報の収集
        if collect_stats:
            stats['trimmed_duration'] = (df_trimmed['timestamp'].max() - df_trimmed['timestamp'].min()) / 1e6

        return df_trimmed, stats

    except Exception as e:
        print(f"Error in trim_based_on_fixations: {str(e)}")
        return df, stats
    
def check_gaze_data_quality(
    df: pd.DataFrame, 
    min_duration: int = 2_000_000,        # 最小録画時間 (2秒)
    min_valid_ratio: float = 0.8,         # 有効なデータの最小比率
    valid_x_range: tuple = (0, 1),        # 有効なx座標の範囲
    valid_y_range: tuple = (0, 1),        # 有効なy座標の範囲
    min_rows: int = 100,                  # 最小行数
    max_gap_duration: int = 100_000,      # 許容される最大ギャップ（100ms）
    expected_sampling_rate: int = 60,      # 期待されるサンプリングレート（Hz）
    min_sampling_rate: int = 30           # 最小許容サンプリングレート（Hz）
) -> Tuple[bool, str]:
    """
    視線データの品質をチェックする関数

    :param df: 視線データのDataFrame
    :param min_duration: 最小録画時間（マイクロ秒）
    :param min_valid_ratio: 有効なデータの最小比率
    :param valid_x_range: 有効なx座標の範囲
    :param valid_y_range: 有効なy座標の範囲
    :param min_rows: 必要な最小行数
    :param max_gap_duration: 許容される最大のタイムスタンプ間隔（マイクロ秒）
    :param expected_sampling_rate: 期待されるサンプリングレート（Hz）
    :param min_sampling_rate: 最小許容サンプリングレート（Hz）
    :return: (データが有効かどうか, 除外理由)
    """
    try:
        # データが空の場合
        if df.empty:
            return False, "Empty data"

        # 行数チェック
        if len(df) < min_rows:
            return False, f"Insufficient data rows: {len(df)} < {min_rows}"

        # 録画時間のチェック
        total_duration = df['timestamp'].max() - df['timestamp'].min()
        if total_duration < min_duration:
            return False, f"Recording too short: {total_duration/1e6:.2f}s"

        # 実際のサンプリングレートを計算
        actual_sampling_rate = len(df) / (total_duration / 1e6)  # Hz
        if actual_sampling_rate < min_sampling_rate:
            return False, f"Low sampling rate: {actual_sampling_rate:.1f}Hz < {min_sampling_rate}Hz"

        # データの連続性チェック
        df_sorted = df.sort_values('timestamp')
        timestamp_gaps = df_sorted['timestamp'].diff().dropna()
        
        # 大きなギャップの検出
        large_gaps = timestamp_gaps[timestamp_gaps > max_gap_duration]
        if not large_gaps.empty:
            gap_count = len(large_gaps)
            max_gap = large_gaps.max() / 1e6  # 秒単位に変換
            return False, f"Found {gap_count} large gaps (max: {max_gap:.2f}s)"

        # データの欠損率を計算
        expected_rows = int((total_duration / 1e6) * expected_sampling_rate)
        missing_data_ratio = 1 - (len(df) / expected_rows)
        if missing_data_ratio > (1 - min_valid_ratio):
            return False, f"High missing data ratio: {missing_data_ratio:.1%}"

        # x, y座標の範囲チェック
        x_cols = ['left_x', 'right_x']
        y_cols = ['left_y', 'right_y']
        
        valid_points = pd.DataFrame()
        for x_col in x_cols:
            valid_points[f'{x_col}_valid'] = df[x_col].between(valid_x_range[0], valid_x_range[1])
        for y_col in y_cols:
            valid_points[f'{y_col}_valid'] = df[y_col].between(valid_y_range[0], valid_y_range[1])

        # データの有効率を計算
        valid_rows = valid_points.all(axis=1)
        valid_ratio = valid_rows.mean()

        if valid_ratio < min_valid_ratio:
            return False, f"Low valid data ratio: {valid_ratio:.2%}"
        
        # method2用の追加チェック：注視点の品質
        fixations = extract_fixations(df)
        if len(fixations) < 3:  # 最低3つの注視点を要求
            return False, f"Too few fixations: {len(fixations)}"

        return True, "Pass"

    except Exception as e:
        return False, f"Error in quality check: {str(e)}"


def extract_fixations(
    df: pd.DataFrame,
    distance_threshold: float = 0.05,
    duration_threshold: int = 150000,  # マイクロ秒単位で表現
    x_cols: List[str] = ['left_x', 'right_x'],
    y_cols: List[str] = ['left_y', 'right_y']
) -> List[Dict[str, Any]]:
    """
    視線データから注視点を抽出する関数

    :param df: 入力のDataFrame
    :param distance_threshold: 注視点とみなす最大距離
    :param duration_threshold: 注視とみなす最小時間（マイクロ秒）
    :param x_cols: x座標の列名リスト
    :param y_cols: y座標の列名リスト
    :return: 注視点のリスト
    """
    if df.empty:
        return []

    # 必要な列が存在するか確認
    required_cols = x_cols + y_cols + ['timestamp']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns. Available columns: {df.columns}")
        return []

    fixations = []
    try:
        # x座標とy座標の平均を計算
        df = df.copy()
        df['x'] = df[x_cols].mean(axis=1)
        df['y'] = df[y_cols].mean(axis=1)

        if len(df) == 0:
            return []

        fixation_start = 0
        fixation_center = np.array([df['x'].iloc[0], df['y'].iloc[0]])
        
        for i in range(1, len(df)):
            current_point = np.array([df['x'].iloc[i], df['y'].iloc[i]])
            distance = np.linalg.norm(current_point - fixation_center)
            
            if distance > distance_threshold:
                duration = df['timestamp'].iloc[i-1] - df['timestamp'].iloc[fixation_start]
                if duration >= duration_threshold:
                    fixations.append({
                        'start_time': df['timestamp'].iloc[fixation_start],
                        'end_time': df['timestamp'].iloc[i-1],
                        'duration': duration,
                        'x': float(fixation_center[0]),
                        'y': float(fixation_center[1])
                    })
                
                fixation_start = i
                fixation_center = current_point
            else:
                fixation_center = (fixation_center * (i - fixation_start) + current_point) / (i - fixation_start + 1)
        
        # 最後の注視点を処理
        if len(df) > fixation_start:
            duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[fixation_start]
            if duration >= duration_threshold:
                fixations.append({
                    'start_time': df['timestamp'].iloc[fixation_start],
                    'end_time': df['timestamp'].iloc[-1],
                    'duration': duration,
                    'x': float(fixation_center[0]),
                    'y': float(fixation_center[1])
                })

    except Exception as e:
        print(f"Error in extract_fixations: {str(e)}")
        return []
    
    return fixations

def preprocess_method1(df: pd.DataFrame) -> np.ndarray:
    # df_trimmed = remove_last_half_second(df)
    # df_trimmed = remove_last_ten_percent(df)
    # df_trimmed = trim_select_button_fixations(df)
    df_trimmed = trim_based_on_fixations(df)

    return df_trimmed[['left_x', 'left_y', 'right_x', 'right_y']].values


def preprocess_method2(df: pd.DataFrame) -> np.ndarray:
    """
    前処理手法2: 注視点を抽出し、生の特徴量を生成する
    正規化はcollate_fnで行うため、ここでは生の値を返す
    
    :param df: 入力のDataFrame
    :return: 注視点の特徴量配列。エラー時は空配列
    """
    try:
        if df.empty:
            return np.array([])

        # df_trimmed = remove_last_half_second(df)
        # df_trimmed = remove_last_ten_percent(df)
        # df_trimmed = trim_select_button_fixations(df)
        df_trimmed = trim_based_on_fixations(df)
        if df_trimmed.empty:
            return np.array([])

        fixations = extract_fixations(df_trimmed)
        if not fixations:
            return np.array([])

        # 注視点データをnumpy配列に変換（生の値のまま）
        fixation_data = np.array([[f['x'], f['y'], f['duration']] for f in fixations])

        

        return fixation_data

    except Exception as e:
        print(f"Error in preprocess_method2: {str(e)}")
        return np.array([])
    
class GazeDataset(Dataset):
    def __init__(self, data: Dict[str, Dict[str, List[Tuple[pd.DataFrame, int]]]], 
                 preprocess_func: str,
                 quality_check_params: Optional[Dict] = None):
        """
        GazeDatasetの初期化
        
        :param data: 生データの辞書
        :param preprocess_func: 前処理関数の指定 ('method1' または 'method2')
        :param quality_check_params: データ品質チェックのパラメータ
        """
        super().__init__()
        self.data = []
        self.labels = []
        self.method = preprocess_func
        self.preprocessing_stats = []  # 前処理の統計情報を保存するリスト
        
        # デフォルトのパラメータを設定
        default_params = {
            'min_duration': 2_000_000,
            'min_valid_ratio': 0.8,
            'valid_x_range': (0, 1),
            'valid_y_range': (0, 1),
            'min_rows': 100,
            'max_gap_duration': 500_000,
            'expected_sampling_rate': 60,
            'min_sampling_rate': 30
        }
        
        self.quality_check_params = {**default_params, **(quality_check_params or {})}
        
        if isinstance(preprocess_func, str):
            if preprocess_func == 'method1':
                self.preprocess_func = preprocess_method1
                print("Using Method 1 preprocessor (4 features)")
            elif preprocess_func == 'method2':
                self.preprocess_func = preprocess_method2
                print("Using Method 2 preprocessor (3 features)")
            else:
                raise ValueError(f"Unknown preprocess_func: {preprocess_func}")
        else:
            raise ValueError("preprocess_func must be 'method1' or 'method2'")

        # 品質統計の初期化
        self.quality_stats = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'failure_reasons': {},
            'sampling_rates': [],
            'durations': []
        }

        self._load_data(data)
        self._validate_and_convert_labels()
        self._print_quality_stats()
        self.print_preprocessing_stats()  # 前処理の統計情報を表示

    def _load_data(self, data):
        """データ読み込み時の処理と統計情報の収集"""
        if not data:
            print("Warning: Empty input data dictionary!")
            return

        for user_type, sessions in data.items():
            for session, data_list in sessions.items():
                for gaze_data, label in data_list:
                    self.quality_stats['total'] += 1
                    
                    try:
                        if gaze_data is None or gaze_data.empty:
                            self.quality_stats['failed'] += 1
                            self.quality_stats['failure_reasons']['Empty data'] = \
                                self.quality_stats['failure_reasons'].get('Empty data', 0) + 1
                            continue

                        # 品質チェック
                        is_valid, reason = check_gaze_data_quality(
                            gaze_data, 
                            **self.quality_check_params
                        )

                        if not is_valid:
                            self.quality_stats['failed'] += 1
                            self.quality_stats['failure_reasons'][reason] = \
                                self.quality_stats['failure_reasons'].get(reason, 0) + 1
                            continue

                        self.quality_stats['passed'] += 1
                        
                        # 基本統計の記録
                        duration = gaze_data['timestamp'].max() - gaze_data['timestamp'].min()
                        sampling_rate = len(gaze_data) / (duration / 1e6)
                        self.quality_stats['sampling_rates'].append(sampling_rate)
                        self.quality_stats['durations'].append(duration)

                        # データの処理と前処理統計の収集
                        trimmed_data, stats = trim_based_on_fixations(gaze_data)
                        stats.update({
                            'session': session,
                            'label': label,
                            'user_type': user_type,
                            'original_samples': len(gaze_data),
                            'sampling_rate': sampling_rate
                        })

                        if self.method == 'method1':
                            if trimmed_data.size > 0:
                                tensor_data = torch.tensor(
                                    trimmed_data[['left_x', 'left_y', 'right_x', 'right_y']].values,
                                    dtype=torch.float32
                                )
                                if tensor_data.dim() == 2 and tensor_data.size(-1) == 4:
                                    self.data.append(tensor_data)
                                    self.labels.append(label)
                                    stats['processed_samples'] = len(tensor_data)
                                    self.preprocessing_stats.append(stats)
                        else:  # method2
                            if not trimmed_data.empty:
                                fixations = extract_fixations(trimmed_data)
                                if fixations:
                                    fixation_data = np.array([[f['x'], f['y'], f['duration']] 
                                                            for f in fixations])
                                    tensor_data = torch.tensor(fixation_data, dtype=torch.float32)
                                    if tensor_data.dim() == 2 and tensor_data.size(-1) == 3:
                                        self.data.append(tensor_data)
                                        self.labels.append(label)
                                        stats['processed_samples'] = len(tensor_data)
                                        self.preprocessing_stats.append(stats)

                    except Exception as e:
                        self.quality_stats['failed'] += 1
                        self.quality_stats['failure_reasons']['Processing error'] = \
                            self.quality_stats['failure_reasons'].get('Processing error', 0) + 1

    def _print_preprocessing_stats(self):
        """前処理の詳細な統計情報を表示"""
        if not self.preprocessing_stats:
            print("\nNo preprocessing statistics available")
            return

        print("\nPreprocessing Statistics Summary:")
        total_samples = len(self.preprocessing_stats)
        
        # トリミングタイプの分布
        trimming_types = {}
        for stats in self.preprocessing_stats:
            t_type = stats['trimming_type']
            trimming_types[t_type] = trimming_types.get(t_type, 0) + 1
        
        print("\nTrimming type distribution:")
        for t_type, count in sorted(trimming_types.items()):
            print(f"{t_type}: {count} samples ({count/total_samples*100:.2f}%)")
        
        # 時間に関する統計
        durations = [(s['original_duration'], s['trimmed_duration']) 
                    for s in self.preprocessing_stats]
        orig_durations, trim_durations = zip(*durations)
        
        print("\nDuration statistics (seconds):")
        print(f"Original  - Mean: {np.mean(orig_durations):.2f}s, "
            f"Std: {np.std(orig_durations):.2f}s")
        print(f"Trimmed   - Mean: {np.mean(trim_durations):.2f}s, "
            f"Std: {np.std(trim_durations):.2f}s")

        # ゼロ除算を防ぐため、元の期間が0より大きい場合のみ削減率を計算
        valid_durations = [(t, o) for t, o in durations if o > 0]
        if valid_durations:
            reduction_rates = [1 - t/o for t, o in valid_durations]
            print(f"Reduction - Mean: {np.mean(reduction_rates)*100:.2f}%")
        else:
            print("Reduction - Cannot calculate (no valid durations)")
        
        # 注視点の統計
        print("\nFixation statistics:")
        total_fix = [s['total_fixations'] for s in self.preprocessing_stats]
        button_fix = [s['button_fixations'] for s in self.preprocessing_stats]
        print(f"Total fixations - Mean: {np.mean(total_fix):.1f}, Std: {np.std(total_fix):.1f}")
        print(f"Button fixations - Mean: {np.mean(button_fix):.1f}, Std: {np.std(button_fix):.1f}")
        
        # ラベルごとの統計
        print("\nStatistics by label:")
        for label in sorted(set(s['label'] for s in self.preprocessing_stats)):
            label_stats = [s for s in self.preprocessing_stats if s['label'] == label]
            print(f"\nLabel {label} ({len(label_stats)} samples):")
            orig_dur = [s['original_duration'] for s in label_stats]
            trim_dur = [s['trimmed_duration'] for s in label_stats]
            total_fix = [s['total_fixations'] for s in label_stats]
            button_fix = [s['button_fixations'] for s in label_stats]
            
            print(f"Duration    - Original: {np.mean(orig_dur):.2f}s, "
                f"Trimmed: {np.mean(trim_dur):.2f}s")

            # ゼロ除算を防ぐため、元の期間が0より大きい場合のみ削減率を計算
            valid_pairs = [(t, o) for t, o in zip(trim_dur, orig_dur) if o > 0]
            if valid_pairs:
                reduction_rates = [1 - t/o for t, o in valid_pairs]
                print(f"Reduction   - {np.mean(reduction_rates)*100:.2f}%")
            else:
                print("Reduction   - Cannot calculate (no valid durations)")

            print(f"Fixations   - Total: {np.mean(total_fix):.1f}, "
                f"Button: {np.mean(button_fix):.1f}")
                    
    def _validate_and_convert_labels(self):
        """
        ラベルを検証し、必要に応じて0-3の範囲に変換する
        """
        if not self.labels:
            print("Warning: No labels to validate")
            return

        unique_labels = sorted(set(self.labels))
        print(f"\nUnique labels before conversion: {unique_labels}")
        
        # 元のラベル分布を表示
        original_label_counts = {}
        for label in self.labels:
            original_label_counts[label] = original_label_counts.get(label, 0) + 1
        print("\nOriginal label distribution:")
        for label, count in sorted(original_label_counts.items()):
            print(f"Original label {label}: {count} samples ({count/len(self.labels)*100:.2f}%)")

        # ラベルの範囲を検出
        min_label = min(unique_labels)
        max_label = max(unique_labels)
        
        # 1-basedの場合は0-basedに変換
        if min_label == 1 and max_label == 4:
            print("Converting from 1-based to 0-based indexing...")
            self.labels = [label - 1 for label in self.labels]
        # 既に0-basedの場合は変換不要
        elif min_label == 0 and max_label == 3:
            print("Labels are already in correct range (0-3)")
        else:
            raise ValueError(f"Unexpected label range: [{min_label}, {max_label}]")

        # 変換後のラベル分布を表示
        converted_label_counts = {}
        for label in self.labels:
            converted_label_counts[label] = converted_label_counts.get(label, 0) + 1
        print("\nConverted label distribution:")
        for label in range(4):  # 0から3まで確実に表示
            count = converted_label_counts.get(label, 0)
            print(f"Label {label}: {count} samples ({count/len(self.labels)*100:.2f}%)")

    def _print_quality_stats(self):
        """品質チェックの統計を詳細に表示（ゼロ除算対策）"""
        print("\nData Quality Statistics:")
        print(f"Total samples processed: {self.quality_stats['total']}")
        
        if self.quality_stats['total'] == 0:
            print("Warning: No data samples were processed!")
            print("Please check the following:")
            print("1. Input data structure")
            print("2. File paths and data loading")
            print("3. Data format and content")
            return

        # データが存在する場合の統計情報
        passed_ratio = (self.quality_stats['passed'] / self.quality_stats['total'] * 100) if self.quality_stats['total'] > 0 else 0
        failed_ratio = (self.quality_stats['failed'] / self.quality_stats['total'] * 100) if self.quality_stats['total'] > 0 else 0
        
        print(f"Passed quality check: {self.quality_stats['passed']} ({passed_ratio:.2f}%)")
        print(f"Failed quality check: {self.quality_stats['failed']} ({failed_ratio:.2f}%)")
        
        if self.quality_stats['sampling_rates']:
            print(f"\nSampling rate statistics:")
            print(f"Mean: {np.mean(self.quality_stats['sampling_rates']):.1f}Hz")
            print(f"Min: {np.min(self.quality_stats['sampling_rates']):.1f}Hz")
            print(f"Max: {np.max(self.quality_stats['sampling_rates']):.1f}Hz")
        
        if self.quality_stats['durations']:
            print(f"\nDuration statistics (seconds):")
            durations_sec = np.array(self.quality_stats['durations']) / 1e6
            print(f"Mean: {np.mean(durations_sec):.2f}s")
            print(f"Min: {np.min(durations_sec):.2f}s")
            print(f"Max: {np.max(durations_sec):.2f}s")
        
        if self.quality_stats['failed'] > 0:
            print("\nFailure reasons:")
            for reason, count in self.quality_stats['failure_reasons'].items():
                ratio = (count / self.quality_stats['failed'] * 100) if self.quality_stats['failed'] > 0 else 0
                print(f"  {reason}: {count} samples ({ratio:.2f}%)")

    def print_preprocessing_stats(self):
        """前処理の詳細な統計情報を表示"""
        if not self.preprocessing_stats:
            print("\nNo preprocessing statistics available")
            return

        print("\nPreprocessing Statistics Summary:")
        total_samples = len(self.preprocessing_stats)
        
        # トリミングタイプの分布
        trimming_types = {}
        for stats in self.preprocessing_stats:
            t_type = stats['trimming_type']
            trimming_types[t_type] = trimming_types.get(t_type, 0) + 1
        
        print("\nTrimming type distribution:")
        for t_type, count in sorted(trimming_types.items()):
            print(f"{t_type}: {count} samples ({count/total_samples*100:.2f}%)")
        
        # 時間に関する統計
        durations = [(s['original_duration'], s['trimmed_duration']) 
                    for s in self.preprocessing_stats]
        orig_durations, trim_durations = zip(*durations)
        
        print("\nDuration statistics (seconds):")
        print(f"Original  - Mean: {np.mean(orig_durations):.2f}s, "
            f"Std: {np.std(orig_durations):.2f}s")
        print(f"Trimmed   - Mean: {np.mean(trim_durations):.2f}s, "
            f"Std: {np.std(trim_durations):.2f}s")

        # ゼロ除算を防ぐため、元の期間が0より大きい場合のみ削減率を計算
        valid_durations = [(t, o) for t, o in durations if o > 0]
        if valid_durations:
            reduction_rates = [1 - t/o for t, o in valid_durations]
            print(f"Reduction - Mean: {np.mean(reduction_rates)*100:.2f}%")
        else:
            print("Reduction - Cannot calculate (no valid durations)")
        
        # 注視点の統計
        print("\nFixation statistics:")
        total_fix = [s['total_fixations'] for s in self.preprocessing_stats]
        button_fix = [s['button_fixations'] for s in self.preprocessing_stats]
        print(f"Total fixations - Mean: {np.mean(total_fix):.1f}, Std: {np.std(total_fix):.1f}")
        print(f"Button fixations - Mean: {np.mean(button_fix):.1f}, Std: {np.std(button_fix):.1f}")
        
        # ラベルごとの統計
        print("\nStatistics by label:")
        for label in sorted(set(s['label'] for s in self.preprocessing_stats)):
            label_stats = [s for s in self.preprocessing_stats if s['label'] == label]
            print(f"\nLabel {label} ({len(label_stats)} samples):")
            orig_dur = [s['original_duration'] for s in label_stats]
            trim_dur = [s['trimmed_duration'] for s in label_stats]
            total_fix = [s['total_fixations'] for s in label_stats]
            button_fix = [s['button_fixations'] for s in label_stats]
            
            print(f"Duration    - Original: {np.mean(orig_dur):.2f}s, "
                f"Trimmed: {np.mean(trim_dur):.2f}s")

            # ゼロ除算を防ぐため、元の期間が0より大きい場合のみ削減率を計算
            valid_pairs = [(t, o) for t, o in zip(trim_dur, orig_dur) if o > 0]
            if valid_pairs:
                reduction_rates = [1 - t/o for t, o in valid_pairs]
                print(f"Reduction   - {np.mean(reduction_rates)*100:.2f}%")
            else:
                print("Reduction   - Cannot calculate (no valid durations)")

            print(f"Fixations   - Total: {np.mean(total_fix):.1f}, "
                f"Button: {np.mean(button_fix):.1f}")
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        return self.data[idx], self.labels[idx]
    
def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    バッチ内のデータをパディングし、マスクを生成する関数。
    method2の場合、duration（3列目）をバッチ内で正規化する。
    
    :param batch: データセットから取得したバッチ
    :return: パディングされたシーケンス、ラベル、長さ、マスク
    """
    if len(batch) == 0:
        raise ValueError("Empty batch received")

    # Sort the batch in descending order of sequence length
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    sequences, labels = zip(*batch)
    
    # Get sequence lengths
    lengths = torch.tensor([seq.shape[0] for seq in sequences])
    
    # Pad sequences
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    
    # Method2の場合（3列目がduration）の正規化
    if padded_sequences.size(-1) == 3:  # method2の特徴数
        # マスクを作成して有効なデータのみを考慮
        valid_mask = torch.arange(padded_sequences.size(1)).expand(len(sequences), -1) < lengths.unsqueeze(1)
        
        # durationデータを取得（3列目）
        durations = padded_sequences[:, :, 2]
        
        # マスクを適用して有効なデータのみを考慮した最大値と最小値を計算
        valid_durations = durations[valid_mask]
        if len(valid_durations) > 0:  # 有効なデータが存在する場合
            min_duration = valid_durations.min()
            max_duration = valid_durations.max()
            
            # 正規化（0除算を防ぐ）
            if max_duration > min_duration:
                normalized_durations = (durations - min_duration) / (max_duration - min_duration)
                padded_sequences[:, :, 2] = normalized_durations
            
    
    # Create mask for attention/padding
    mask = torch.arange(padded_sequences.size(1)).expand(len(sequences), -1) < lengths.unsqueeze(1)
    
    # Convert labels to long tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_sequences, labels, lengths, mask

def create_data_loaders(dataset: GazeDataset, batch_size: int, train_ratio: float = 0.8):
    """
    データセットからtrain_loaderとtest_loaderを作成する

    :param dataset: GazeDatasetインスタンス
    :param batch_size: バッチサイズ
    :param train_ratio: 訓練データの割合
    :return: (train_loader, test_loader)のタプル
    """
    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError("Dataset is empty")

    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    print(f"\nSplitting dataset:")
    print(f"Total size: {dataset_size}")
    print(f"Train size: {train_size}")
    print(f"Test size: {test_size}")

    # データセットを分割（シード値を固定）
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, test_loader


def analyze_sequence_lengths(dataset: GazeDataset) -> Dict[str, Dict[str, float]]:
    """
    データセットのシーケンス長を分析する関数
    
    :param dataset: 分析対象のGazeDataset
    :return: 統計情報を含む辞書
    """
    lengths = [data.shape[0] for data in dataset.data]
    
    stats = {
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'median': np.median(lengths),
        'q1': np.percentile(lengths, 25),
        'q3': np.percentile(lengths, 75),
        'total_sequences': len(lengths)
    }
    
    return stats

def print_sequence_stats(stats: Dict[str, Dict[str, float]], method: str):
    """
    シーケンス長の統計情報を表示
    
    :param stats: 統計情報を含む辞書
    :param method: 分析対象のメソッド名
    """
    print(f"\nSequence Length Analysis for {method}:")
    print(f"Total sequences: {stats['total_sequences']}")
    print(f"Mean length: {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"Median length: {stats['median']:.2f}")
    print(f"Min length: {stats['min']:.2f}")
    print(f"Max length: {stats['max']:.2f}")
    print(f"Q1 (25th percentile): {stats['q1']:.2f}")
    print(f"Q3 (75th percentile): {stats['q3']:.2f}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from load_data import load_gaze_data
    
    # データの読み込み
    raw_data = load_gaze_data("/workspace/data")
    
    # 品質チェックパラメータの設定
    quality_params = {
        'min_duration': 2_000_000,
        'min_valid_ratio': 0.8,
        'min_rows': 100,
        'max_gap_duration': 500_000,
        'expected_sampling_rate': 60,
        'min_sampling_rate': 30
    }
    
    # Method1とMethod2のデータセット作成
    dataset_method1 = GazeDataset(raw_data, preprocess_func='method1', quality_check_params=quality_params)
    dataset_method2 = GazeDataset(raw_data, preprocess_func='method2', quality_check_params=quality_params)
    
    # シーケンス長の分析
    stats_method1 = analyze_sequence_lengths(dataset_method1)
    stats_method2 = analyze_sequence_lengths(dataset_method2)
    
    # 統計情報の表示
    print_sequence_stats(stats_method1, "Method 1 (Raw Gaze Data)")
    print_sequence_stats(stats_method2, "Method 2 (Fixation Data)")
    
    # シーケンス長の分布をプロット
    plt.figure(figsize=(15, 5))
    
    # Method1のヒストグラム
    plt.subplot(121)
    lengths_method1 = [data.shape[0] for data in dataset_method1.data]
    plt.hist(lengths_method1, bins=50, alpha=0.7)
    plt.title("Method 1: Raw Gaze Data Sequence Lengths")
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    
    # Method2のヒストグラム
    plt.subplot(122)
    lengths_method2 = [data.shape[0] for data in dataset_method2.data]
    plt.hist(lengths_method2, bins=50, alpha=0.7)
    plt.title("Method 2: Fixation Data Sequence Lengths")
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig('sequence_length_distribution.png')
    plt.close()
    
    # 基本的な比較統計
    print("\nComparison Statistics:")
    avg_reduction = (stats_method1['mean'] - stats_method2['mean']) / stats_method1['mean'] * 100
    print(f"Average sequence length reduction: {avg_reduction:.2f}%")
    print(f"Method 1 to Method 2 ratio (mean): {stats_method2['mean']/stats_method1['mean']:.3f}")
    
    # 相関分析
    if len(dataset_method1.data) == len(dataset_method2.data):
        lengths1 = np.array([data.shape[0] for data in dataset_method1.data])
        lengths2 = np.array([data.shape[0] for data in dataset_method2.data])
        correlation = np.corrcoef(lengths1, lengths2)[0, 1]
        print(f"Correlation between sequence lengths: {correlation:.3f}")
        
        # 散布図
        plt.figure(figsize=(8, 8))
        plt.scatter(lengths1, lengths2, alpha=0.5)
        plt.plot([0, max(lengths1)], [0, max(lengths1)], 'r--', alpha=0.5)  # 対角線
        plt.title("Sequence Lengths: Method 1 vs Method 2")
        plt.xlabel("Method 1 Sequence Length")
        plt.ylabel("Method 2 Sequence Length")
        plt.savefig('sequence_length_correlation.png')
        plt.close()
