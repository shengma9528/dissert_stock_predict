import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import MinMaxScaler
from vmdpy import VMD
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Conv1D, MaxPooling1D, Dropout, Bidirectional, GRU
from tensorflow.keras.optimizers import Adam

class SVMD:
    def __init__(self, alpha=2000, tau=0, K=8, DC=0, init=1, tol=1e-7, stop_tol=1e-7, fix_K=False):
        self.alpha = alpha; self.tau = tau; self.K = K; self.DC = DC
        self.init = init; self.tol = tol; self.stop_tol = stop_tol
        self.fix_K = fix_K

    def __call__(self, data):
        data = np.asarray(data).flatten()
        residual = data.copy()
        original_var = np.var(data)
        u_list, omega_list = [], []
        if original_var < 1e-10:
            return np.array([data]), None, np.array([[0.0]])
        for k in range(self.K):
            if (np.var(residual) / original_var) < self.stop_tol:
                break
            if k == self.K - 1:
                if self.fix_K:
                    u_list.append(residual)
                    omega_list.append(0.5)
                else:
                    u, _, omega = VMD(residual, self.alpha, self.tau, 1, self.DC, self.init, self.tol)
                    u_list.append(u[0]); omega_list.append(omega[-1, 0])
            else:
                u, _, omega = VMD(residual, self.alpha, self.tau, 2, self.DC, self.init, self.tol)
                freqs = omega[-1, :]
                idx = np.argmin(freqs)
                u_list.append(u[idx]); omega_list.append(freqs[idx])
                residual = residual - u[idx]
        if not u_list:
            u_list = [data]; omega_list = [0.0]

        if self.fix_K:
            while len(u_list) < self.K:
                u_list.append(np.zeros_like(data))
                omega_list.append(0.0)

        return np.array(u_list), None, np.array([omega_list])


class ComparePredictor:
    def __init__(self, ticker='TSLA', start_date='2012-01-01', end_date='2024-01-01'):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        # 加载数据
        self._get_stock_data()

    def _get_stock_data(self):
        cache_dir = "./"
        os.makedirs(cache_dir, exist_ok=True)
        file_path = os.path.join(cache_dir, f"{self.ticker}_{self.start_date}_{self.end_date}.csv")
        if os.path.exists(file_path):
            self.raw_data = pd.read_csv(file_path)
        else:
            data = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            self.raw_data = data.reset_index()[["Date", "Close", "High", "Low", "Open", "Volume"]]
            self.raw_data.to_csv(file_path, index=False)
        self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'])
        self.close_prices = self.raw_data[['Close']].values
        vol_log = np.log1p(self.raw_data[['Volume']].values)
        vol_diff = np.diff(vol_log, axis=0, prepend=vol_log[0].reshape(1,1))
        vol_energy = pd.Series(vol_log.flatten()).rolling(window=5).std().fillna(0).values.reshape(-1, 1)
        self.volumes = vol_diff * (1 + vol_energy)

    def _get_features_close(self, time_step):
        # 1. 计算边界
        total_len = len(self.close_prices) - time_step
        train_end = int(total_len * 0.8)
        val_end = int(total_len * 0.9)

        x_arr, y_arr = [], []
        scalers = []

        # 确保 close_prices 是 numpy 数组并重塑为 (-1, 1) 方便 Scaler 处理
        # 如果 self.close_prices 是 Series，这里用 .values
        prices = np.array(self.close_prices).reshape(-1, 1)

        for i in range(len(prices) - time_step):
            # 提取窗口数据 (time_step, 1)
            datax = prices[i : i + time_step]
            # 提取目标点 (1, 1)
            datay = prices[i + time_step].reshape(1, -1)

            # 实例化并拟合
            scaler_curr = MinMaxScaler(feature_range=(0, 1))
            # fit_transform 期望 2D 输入，返回 2D
            win_curr_x_scaled = scaler_curr.fit_transform(datax).flatten()

            # transform 同样期望 2D 输入
            # 直接传入 datay (已经是 1x1 的 2D 数组)
            win_curr_y_scaled = scaler_curr.transform(datay)[0][0]

            x_arr.append(win_curr_x_scaled)
            y_arr.append(win_curr_y_scaled)
            scalers.append(scaler_curr)

        X = np.array(x_arr)
        y = np.array(y_arr)

        # 2. 切分数据集 (修复之前 X_val 未定义的错误)
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        # 对应的 Scaler 也需要切分，确保 test 阶段能用到对应的 scaler
        scaler_test = scalers[val_end:]

        # 3. 转换形状以适配 PyTorch (样本数, 时间步, 特征数)
        X_train = X_train.reshape(X_train.shape[0], time_step, 1)
        X_val = X_val.reshape(X_val.shape[0], time_step, 1)
        X_test = X_test.reshape(X_test.shape[0], time_step, 1)
        # print(f"X_train:{X_train.shape}  X_val:{X_val.shape}  X_test:{X_test.shape}")

        # 确保返回所有 7 个变量（匹配你调用处的解包）
        return X_train, y_train, X_val, y_val, X_test, y_test, scaler_test

    def _evaluate_metrics(self, y_true, y_pred, model_name):
        # print(f"_evaluate_metrics  y_true.shape: {y_true.shape}, y_pred.shape: {y_pred.shape}")

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"[{model_name}] 真实空间评估 -> MAE: {mae:.4f} | MAPE: {mape:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
        return mae, mse, r2

    def test_LSTM(self, time_step=20, h1=64, h2=32, lr=0.001):
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = self._get_features_close(time_step)

        model = Sequential([
            Input(shape=(time_step, 1)),
            Bidirectional(LSTM(int(h1), activation='tanh', return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(int(h2), activation='tanh')),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        preds = model.predict(X_test, verbose=0).flatten()

        preds_real = np.array([
            s.inverse_transform([[p]])[0][0] for s, p in zip(scaler, preds)
        ])

        y_test_real = np.array([
            s.inverse_transform([[y]])[0][0] for s, y in zip(scaler, y_test)
        ])

        self._evaluate_metrics(y_test_real, preds_real, "Pure LSTM")

    def test_LightGBM(self, time_step=20):
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = self._get_features_close(time_step)

        # LightGBM 接受 2D 输入 (samples, features)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        # model = LGBMRegressor(n_estimators=20, learning_rate=0.05, max_depth=5, random_state=42)
        model = LGBMRegressor(
            n_estimators=20,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbose=-1,              # 核心修改：关闭 LightGBM 的内部打印
            force_col_wise=True      # 核心修改：消除 overhead 提示
        )
        model.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)])

        preds = model.predict(X_test_flat)

        preds_real = np.array([s.inverse_transform([[p]])[0][0] for s, p in zip(scaler, preds)])
        y_test_real = np.array([s.inverse_transform([[y]])[0][0] for s, y in zip(scaler, y_test)])

        self._evaluate_metrics(y_test_real, preds_real, "Pure LightGBM")

    def test_CNN_LSTM(self, time_step=20, h1=64, h2=32, lr=0.001):
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = self._get_features_close(time_step)

        model = Sequential([
            Input(shape=(time_step, 1)),
            Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            LSTM(int(h1), activation='tanh', return_sequences=True),
            Dropout(0.2),
            LSTM(int(h2), activation='tanh'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        preds = model.predict(X_test, verbose=0).flatten()

        preds_real = np.array([
            s.inverse_transform([[p]])[0][0] for s, p in zip(scaler, preds)
        ])

        y_test_real = np.array([
            s.inverse_transform([[y]])[0][0] for s, y in zip(scaler, y_test)
        ])

        self._evaluate_metrics(y_test_real, preds_real, "CNN-LSTM (No Decomp)")

    def prepare_svmd_data(self, time_step=20, K=3):
        if time_step % 2 == 1:
            time_step += 1

        total_samples = len(self.close_prices) - time_step
        train_end = int(total_samples * 0.8)
        val_end = int(total_samples * 0.9)

        x_arr = []
        scalers_min, scalers_max = [], []
        real_test_data = []

        for i in range(total_samples):
            datax  = self.close_prices[i : i + time_step].reshape(1, -1)
            # print("datax:", datax[0, -1])
            real_test_data.append(datax[0, -1])
            tmp, d_min, d_max    = self._row_normalize(datax)

            svmd  = SVMD(K=K)
            u, _, omega = svmd(tmp)
            # if u.shape[0] != K:
            #     print("datax", datax)
            #     print("tmp", tmp)

            # if i == 3:
            #   print(f"u: {u.shape}")
            #   print(f"tmp: {tmp}")
            #   print(f"u: {u}")
            x_arr.append(u)
            scalers_min.append(d_min)
            scalers_max.append(d_max)

        X = np.array(x_arr)
        real_test_data = np.array(real_test_data).reshape(-1, 1)

        scalers_min = np.array(scalers_min).reshape(-1, 1)
        scalers_max = np.array(scalers_max).reshape(-1, 1)


        # print(f"scalers_max: {scalers_max.shape}")
        # print(f"X: {X[2, :, :]}")

        X_train = X[:train_end, :, :]
        X_val   = X[train_end:val_end, :, :]
        X_test  = X[val_end:, :, :]
        scalers_min = scalers_min[val_end:, :]
        scalers_max = scalers_max[val_end:, :]
        real_test_data = real_test_data[val_end:]
        # print("real_test_data:", real_test_data.shape)

        # print(f"X: {X.shape}  X_train:{X_train.shape}  X_val:{X_val.shape}  X_test:{X_test.shape} scalers_min:{scalers_min.shape} scalers_min:{scalers_max.shape}")

        return X_train, X_val, X_test, scalers_min, scalers_max, real_test_data

    def _row_normalize(self, data):
        d_min = data.min(axis=1, keepdims=True)
        d_max = data.max(axis=1, keepdims=True)
        d_scaled = (data - d_min) / (d_max - d_min + 1e-8)
        return d_scaled, d_min, d_max

    def _row_inverse_normalize(self, data, t_min, t_max):
        return data * (t_max - t_min) + t_min

    def test_VMD_LSTM(self, time_step=10):
        K       = 5
        h1, h2, lr = 64, 32, 1e-3

        # ✅ 接收 test_scalers
        X_train, X_val, X_test, t_min, t_max, real_test_data = \
            self.prepare_svmd_data(time_step, K)
        # print(f"tr:{X_train.shape}, X_v:{X_val.shape}, X_t:{X_test.shape}, t_min:{t_min.shape}, t_max:{t_max.shape}")
        preds = []

        for i in range(K):
            model = Sequential([
                Input(shape=(time_step - 1, 1)),
                Bidirectional(LSTM(int(h1), activation='tanh', return_sequences=True)),
                Dropout(0.2),
                Bidirectional(LSTM(int(h2), activation='tanh')),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

            X_tr = X_train[:, i, :]
            y_tr = X_tr[:, -1:]
            y_tr = y_tr[1:, :]
            X_tr = X_tr[:-1, :]

            X_v = X_val[:, i, :]
            y_v = X_v[:, -1:]
            y_v = y_v[1:, :]
            X_v = X_v[:-1, :]

            X_t = X_test[:, i, :]
            y_t = X_t[:, -1:]
            y_t = y_t[1:, :]
            X_t = X_t[:-1, :]

            # print(f"tr:{X_tr.shape}, X_v:{X_v.shape}, X_t:{X_t.shape}")
            # ✅ 此时喂给模型的是纯净的、无量纲干扰的归一化数据
            model.fit(X_tr, y_tr, epochs=10, batch_size=32, verbose=0,
                    validation_data=(X_v, y_v))

            # y_pred = model.predict(X_v)
            # val_r2 = r2_score(y_v, y_pred)
            # print(f"验证集 K:{i} R2: {val_r2:.4f}")

            pred = model.predict(
                X_t, verbose=0
            )
            # print(f"pred.shape: {pred.shape}")
            preds.append(pred)


        t_min = t_min[1:, :]
        t_max = t_max[1:, :]
        real_test_data = real_test_data[1:, :]
        # print("real_test_data:", real_test_data.shape)

        preds = np.array(preds)
        # print(f"preds.shape: {preds.shape}")
        preds_real = np.sum(preds, axis=0)
        preds_real = self._row_inverse_normalize(preds_real, t_min, t_max)

        self._evaluate_metrics(real_test_data, preds_real, "VMD-LSTM (Strict)")

    def test(self):
        pass

if __name__ == "__main__":
    predictor = ComparePredictor()
    predictor.test_LSTM()
    predictor.test_LightGBM()
    predictor.test_CNN_LSTM()
    predictor.test_VMD_LSTM()
    # predictor.test()