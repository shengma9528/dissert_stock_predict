import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os
import time
import warnings
from datetime import datetime
from vmdpy import VMD
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Conv1D, MaxPooling1D, Dropout, Bidirectional, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K  
# XGBoost
from xgboost import XGBClassifier, XGBRegressor
# LightGBM
from lightgbm import LGBMClassifier, LGBMRegressor


warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
tf.random.set_seed(42)

LOG_FILENAME = f"data/log/report_SMVD_LSTM_30.txt"
FINANCE_DIR = "data/finance"
CACHE_DIR = "data/cache"

# LOG_FILENAME = f"./report_SMVD_LSTM_20.txt"
# FINANCE_DIR = "./"


# ================= SVMD =================
class SVMD:
    def __init__(self, alpha=2000, tau=0, K=8, DC=1, init=1, tol=1e-7, stop_tol=1e-7):
        self.alpha = alpha; self.tau = tau; self.K = K; self.DC = DC
        self.init = init; self.tol = tol; self.stop_tol = stop_tol

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
        return np.array(u_list), None, np.array([omega_list])

# ================= DBO =================
class DBO:
    def __init__(self, obj_func, dim=5, pop_size=10, max_iter=10, lb=None, ub=None):
        self.obj_func = obj_func; self.dim = dim
        self.pop_size = pop_size; self.max_iter = max_iter
        self.lb = np.array(lb); self.ub = np.array(ub)
        self.X = np.random.uniform(self.lb, self.ub, (pop_size, dim))
        self.fitness = np.array([obj_func(x) for x in self.X])
        self.best_x = self.X[np.argmin(self.fitness)].copy()
        self.best_fitness = np.min(self.fitness)

    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.pop_size):
                r2 = np.random.rand()
                if i < self.pop_size * 0.2:
                    if r2 < 0.9:
                        a = 1.0 if np.random.rand() > 0.5 else -1.0
                        self.X[i] = self.X[i] + a * 0.2 * np.abs(self.X[i] - self.best_x)
                    else:
                        self.X[i] = self.X[i] + np.tan(np.random.rand() * np.pi/2) * 0.1
                elif i < self.pop_size * 0.5:
                    self.X[i] = self.best_x + np.random.randn(self.dim) * (self.ub - self.lb) * 0.1
                self.X[i] = np.clip(self.X[i], self.lb, self.ub)
                fit = self.obj_func(self.X[i])
                if fit < self.best_fitness:
                    self.best_fitness = fit; self.best_x = self.X[i].copy()
        return self.best_x

# ================= KELM =================
class KELM:
    def __init__(self, C=1000, gamma=None):
        self.C = C; self.gamma = gamma
        self.X_train = None; self.beta = None

    def fit(self, X, y):
        self.X_train = X
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1] if X.shape[1] > 0 else 1.0
        Omega = rbf_kernel(X, X, gamma=self.gamma)
        I = np.eye(len(X))
        self.beta = np.linalg.inv(Omega + I / self.C).dot(y)

    def predict(self, X):
        Omega_test = rbf_kernel(X, self.X_train, gamma=self.gamma)
        return Omega_test.dot(self.beta)

# ================= 预测策略主类 =================
class VMDStrategyThreePredictor:
    def __init__(self, 
                ticker="AAPL", start_date="2010-01-01", end_date="2025-01-01",
                K=8, alpha=2000, train_size_ratio=0.75, val_size_ratio=0.15, time_step=30, rf_estimator=60,
                 show_plt=False, kl_flag=0, km_flag=0, enable_dbo=0, enable_kelm=True,
                 merged_K=8, trend_imfs=None, mid_imfs=None, noise_imfs=None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.K = K; 
        self.kl_flag = kl_flag
        self.km_flag = km_flag
        self.enable_dbo = enable_dbo
        self.enable_kelm = enable_kelm
        self.merged_K = merged_K
        self.trend_imfs = trend_imfs if trend_imfs is not None else [0]
        self.mid_imfs   = mid_imfs   if mid_imfs   is not None else [1, 2, 3]
        self.noise_imfs = noise_imfs if noise_imfs is not None else [4, 5, 6, 7]
        self.alpha = alpha; 
        self.train_size_ratio = train_size_ratio
        self.val_size_ratio = val_size_ratio
        
        # 确保传入 VMD 的基础 time_step 始终为偶数
        if time_step % 2 == 1:
            time_step = time_step + 1
        self.time_step = time_step
        
        self.rf_estimator = rf_estimator; 
        self.show_plt = show_plt

    def _get_stock_data(self, ticker, start_date, end_date):
        os.makedirs(FINANCE_DIR, exist_ok=True)
        file_path = os.path.join(FINANCE_DIR, f"{ticker}_{start_date}_{end_date}.csv")
        raw_data = None
        if os.path.exists(file_path):
            raw_data = pd.read_csv(file_path)
        else:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            raw_data = data.reset_index()[["Date", "Close", "High", "Low", "Open", "Volume"]]
            raw_data.to_csv(file_path, index=False)
        raw_data['Date'] = pd.to_datetime(raw_data['Date'])
        close_prices = raw_data[['Close']].values
        volumes = np.log1p(raw_data[['Volume']].values)
        return close_prices, volumes

    def _vmd_decompose(self, data_1d):
        local_mean = np.mean(data_1d)
        data_zero_mean = data_1d - local_mean
        svmd = SVMD(alpha=self.alpha, tau=0, K=self.K, DC=0, init=1, tol=1e-7)
        u, _, omega = svmd(data_zero_mean)
        
        center_freqs = omega[0]
        valid_indices = np.where(center_freqs >= 0.0)[0]
        u = u[valid_indices]; 
        center_freqs = center_freqs[valid_indices]
        num_target_components = self.merged_K
        freqs_2d = center_freqs.reshape(-1, 1)
        n_clusters = min(len(freqs_2d), num_target_components)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(freqs_2d)
        cluster_centers = kmeans.cluster_centers_.flatten()
        sorted_cluster_indices = np.argsort(cluster_centers)
        
        u_merged = []
        for i, cluster_idx in enumerate(sorted_cluster_indices):
            component_indices = np.where(labels == cluster_idx)[0]
            merged_signal = np.sum(u[component_indices], axis=0)
            if i == 0:
                merged_signal = merged_signal + local_mean
            u_merged.append(merged_signal)
        while len(u_merged) < num_target_components:
            u_merged.append(np.zeros_like(data_1d))
            
        res = np.array(u_merged)
        
        return res  

    def _build_lstm_model_1(self, h1=64, h2=32, lr=0.001):
        model = Sequential([
            Input(shape=(self.time_step, 2)),
            LSTM(int(h1), activation='tanh', return_sequences=True),
            Dropout(0.2),
            LSTM(int(h2), activation='tanh'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model
    
    def _build_bilstm_model_3(self, h1=64, h2=32, lr=0.001):
        model = Sequential([
            Input(shape=(self.time_step - 1, 2)),
            Bidirectional(LSTM(int(h1), activation='tanh', return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(int(h2), activation='tanh')),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model

    def _build_gru_model_5(self, h1=64, h2=32, lr=0.001):
        model = Sequential([
            Input(shape=(self.time_step - 1, 2)),
            Bidirectional(GRU(int(h1), activation='tanh', return_sequences=True)),
            Dropout(0.2),
            Bidirectional(GRU(int(h2), activation='tanh')),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model

    def _build_cnn_lstm_model_2(self, h1=64, h2=32, lr=0.001):
        model = Sequential([
            Input(shape=(self.time_step - 1, 2)),
            Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            LSTM(int(h1), activation='tanh', return_sequences=True),
            Dropout(0.2),
            LSTM(int(h2), activation='tanh'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model

    def _get_prepared_data(self, ticker, start_date, end_date):
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_filename = f"{self.ticker}_{self.start_date}_{self.end_date}_K{self.K}_ts{self.time_step}_mergeK{self.merged_K}.npz"
        cache_path = os.path.join(CACHE_DIR, cache_filename)
        if os.path.exists(cache_path):
            print(f"📦 发现数据缓存，正在直接加载: {cache_filename}")
            loaded = np.load(cache_path)
            
            # 恢复局部变量
            X_train = loaded['X_train']
            X_val   = loaded['X_val']
            X_test  = loaded['X_test']
            V_train = loaded['V_train']
            V_val   = loaded['V_val']
            V_test  = loaded['V_test']

            self.V_tr_v_data = loaded['V_tr_v_data']
            self.X_tr_v_data = loaded['X_tr_v_data']
            self.scalers_tr_v_min = loaded['scalers_tr_v_min'] 
            self.scalers_tr_v_max = loaded['scalers_tr_v_max']
            self.real_tr_v_data = loaded['real_tr_v_data']
            
            # 恢复类的全局(self)变量
            self.scalers_test_min = loaded['scalers_test_min']
            self.scalers_test_max = loaded['scalers_test_max']
            self.real_test_data   = loaded['real_test_data']
            
            print(f"✅ 从缓存加载成功！X_train 形状: {X_train.shape}")
            return X_train, X_val, X_test, V_train, V_val, V_test
        else:
            close_prices, volumes = self._get_stock_data(self.ticker, self.start_date, self.end_date)
            X_train, X_val, X_test, V_train, V_val, V_test = self._prepare_data(close_prices, volumes)
            # 预处理完成，正在写入缓存
            np.savez_compressed(
                cache_path,
                X_train=X_train, X_val=X_val, X_test=X_test,
                V_train=V_train, V_val=V_val, V_test=V_test,

                scalers_test_min=self.scalers_test_min,
                scalers_test_max=self.scalers_test_max,
                real_test_data=self.real_test_data,

                V_tr_v_data=self.V_tr_v_data,
                X_tr_v_data=self.X_tr_v_data,
                scalers_tr_v_min=self.scalers_tr_v_min, 
                scalers_tr_v_max=self.scalers_tr_v_max,
                real_tr_v_data=self.real_tr_v_data
            )
            # 缓存写入成功！后续运行将秒进模型训练阶段
        return X_train, X_val, X_test, V_train, V_val, V_test

    def _prepare_data(self, close_prices, volumes):
        # 处理数据
        total_len = len(close_prices) - self.time_step
        real_data = []
        x_arr, v_arr = [], []
        scalers_min, scalers_max = [], []
        for i in range(total_len):
            win_x = close_prices[i : i + self.time_step].reshape(1, -1)
            real_data.append(win_x[0, -1])
            tmp_x, d_min, d_max = self._row_normalize(win_x)
            imfs_x = self._vmd_decompose(tmp_x) 

            x_arr.append(imfs_x)  
            scalers_min.append(d_min)
            scalers_max.append(d_max)

            win_v = volumes[i : i + self.time_step]
            tmp_v, _, _ = self._row_normalize(win_v.reshape(1, -1))
            v_arr.append(tmp_v)
        x_arr = np.array(x_arr)
        v_arr = np.array(v_arr)
        
        train_end = int(total_len * self.train_size_ratio)
        val_end = int(train_end + total_len * self.val_size_ratio)
        # print(f"total_len: {total_len}, train_end: {train_end}, val_end: {val_end}")
        
        real_data = np.array(real_data).reshape(-1, 1)
        scalers_min = np.array(scalers_min).reshape(-1, 1)
        scalers_max = np.array(scalers_max).reshape(-1, 1)   
        
        X_train = x_arr[:train_end, :, :] 
        X_val   = x_arr[train_end:val_end, :, :] 
        X_test  = x_arr[val_end:, :, :]
        V_train = v_arr[:train_end, :, :] 
        V_val   = v_arr[train_end:val_end, :, :] 
        V_test  = v_arr[val_end:, :, :]
        self.V_tr_v_data = v_arr[:val_end, :, :]
        self.X_tr_v_data = x_arr[:val_end, :, :]
        self.scalers_tr_v_min = scalers_min[:val_end, :] 
        self.scalers_tr_v_max = scalers_max[:val_end, :]
        self.scalers_test_min = scalers_min[val_end:, :] 
        self.scalers_test_max = scalers_max[val_end:, :]
        # print(f"_prepare_data X_train.shape: {X_train.shape}, X_val.shape: {X_val.shape}, X_test: {X_test.shape}, V_train.shape: {V_train.shape}, V_val.shape: {V_val.shape}, V_test: {V_test.shape}")

        # 用于测试
        self.real_test_data = real_data[val_end:]
        # 用于验证集
        self.real_tr_v_data = real_data[:val_end]

        return X_train, X_val, X_test, V_train, V_val, V_test
        
    def _row_normalize(self, data):
        d_min = data.min(axis=1, keepdims=True)
        d_max = data.max(axis=1, keepdims=True)
        d_scaled = (data - d_min) / (d_max - d_min + 1e-8)
        return d_scaled, d_min, d_max
    
    def _row_inverse_normalize(self, data, t_min, t_max):
        return data * (t_max - t_min + 1e-8) + t_min

    def _get_dl_model(self, h1=64, h2=32, lr=0.001):
        if self.kl_flag == 2:
            return self._build_cnn_lstm_model_2(h1, h2, lr)
        elif self.kl_flag == 3:
            return self._build_bilstm_model_3(h1, h2, lr)
        elif self.kl_flag == 5:
            return self._build_gru_model_5(h1, h2, lr)
        return self._build_lstm_model_1(h1, h2, lr)

    def _get_ml_model(self, n_estimators=50):
        if self.km_flag == 2:
            return LGBMRegressor(n_estimators=n_estimators)
        elif self.km_flag == 3:
            return XGBRegressor(n_estimators=n_estimators)
        return RandomForestRegressor(n_estimators=n_estimators)

    # 获取第 i 个 IMF 的数据
    def _get_Y(self, X, i, return_y=False):
        X_tr = X[:, i, :]
        y_tr = None
        if return_y:
            y_tr = X_tr[:, -1:]
            y_tr = y_tr[1:, :]
        X_tr = X_tr[:-1, :]
        return X_tr, y_tr

    def _train_models(self, X_train, X_val, V_train, V_val, dl_pool_size=8, dl_max_iter=8, n_estimators=50):
        self.trained_models = {}
        V_train = V_train[:-1, -1, :]
        V_val = V_val[:-1, -1, :]
        # print(f"train_models X_train.shape: {X_train.shape}, X_val.shape: {X_val.shape}, V_train.shape: {V_train.shape}, V_val.shape: {V_val.shape}")
        
        for k in range(self.merged_K):
            X_tr, y_tr = self._get_Y(X_train, k, return_y=True)
            X_v, y_v = self._get_Y(X_val, k, return_y=True)
            # print(f"k: {k}, X_tr.shape: {X_tr.shape}, y_tr.shape: {y_tr.shape}, X_v.shape: {X_v.shape}, y_v.shape: {y_v.shape}")

            if k in self.trend_imfs:
                model = LinearRegression()
                model.fit(X_tr, y_tr)
                self.trained_models[k] = model

            elif k in self.mid_imfs:
                X_lstm = np.stack((X_tr, V_train), axis=-1)
                X_v_lstm = np.stack((X_v, V_val), axis=-1)

                # 1. 定义 EarlyStopping
                early_stopping = EarlyStopping(
                    monitor='val_loss',        # 监控指标：验证集误差
                    patience=10,               # 容忍度：连续10个epoch没有改善就停止
                    restore_best_weights=True  # 核心设置：训练停止后，自动将模型权重回滚到历史表现最好的一轮
                )

                best_h1, best_h2, best_lr, best_epochs, best_batch_size = 64, 32, 0.005, 10, 32

                model = self._get_dl_model(best_h1, best_h2, best_lr)
                model.fit(X_lstm, y_tr, epochs=best_epochs, batch_size=best_batch_size, verbose=0, validation_data=(X_v_lstm, y_v), callbacks=[early_stopping])
                self.trained_models[k] = model
            else:
                model = self._get_ml_model(n_estimators=n_estimators)
                model.fit(X_tr, y_tr)
                self.trained_models[k] = model
        
        if self.enable_kelm:
            preds = []
            V_val = self.V_tr_v_data[:-1, -1, :]
            for k in range(self.merged_K):
                X_v, y_v = self._get_Y(self.X_tr_v_data, k, return_y=True)
                
                if k in self.trend_imfs:
                    pred = self.trained_models[k].predict(X_v)
                elif k in self.mid_imfs:
                    # print(f"k: {k}, X.shape: {X.shape}, v_tr_v_data.shape: {v_tr_v_data.shape}")
                    X_V_combined = np.stack((X_v, V_val), axis=-1)
                    pred = self.trained_models[k].predict(X_V_combined)
                else:
                    pred = self.trained_models[k].predict(X_v)
                pred = pred.reshape(-1, 1)
                preds.append(pred)

                component_r2 = r2_score(y_v, pred)
                reprot = f"val IMF-{k}: R2 = {component_r2:.4f}"
                print(reprot)
                with open(LOG_FILENAME, 'a', encoding='utf-8') as f:
                    f.write(reprot + "\n")
            
            t_min = self.scalers_tr_v_min[1:, :]
            t_max = self.scalers_tr_v_max[1:, :]
            tr_v_data = self.real_tr_v_data[1:, :]
            # print("real_test_data:", real_test_data.shape)

            preds = np.array(preds)
            # print(f"preds.shape: {preds.shape}")
            preds_real = np.sum(preds, axis=0)
            preds_real = self._row_inverse_normalize(preds_real, t_min, t_max)

            error_train = tr_v_data - preds_real
            # X_tr_v_data = self.X_tr_v_data[:-1, :]
            X_v_slice = self.X_tr_v_data[:-1, :, :] 
            X_v_data = X_v_slice.reshape(X_v_slice.shape[0], -1)

            self.kelm_model = KELM(C=100, gamma=0.1)
            self.kelm_model.fit(X_v_data, error_train)

    def _predict_test_data(self, X_test, V_test):
        preds = []
        v_test = V_test[:-1, -1, :]
        for k in range(self.merged_K):
            # print(f"k: {k}  predict_test_data X_test: {X_test.shape}")
            X_t, y_t = self._get_Y(X_test, k, return_y=True)

            if k in self.trend_imfs:
                pred = self.trained_models[k].predict(X_t)
            elif k in self.mid_imfs:
                # print(f"k: {k}  predict_test_data X_t: {X_t.shape}  v_test: {v_test.shape} ")
                X_input = np.stack((X_t, v_test), axis=-1)
                pred = self.trained_models[k].predict(X_input, verbose=0)
            else:
                pred = self.trained_models[k].predict(X_t)
            pred = pred.reshape(-1, 1)
            preds.append(pred)

            component_r2 = r2_score(y_t, pred)
            reprot = f"test IMF-{k}: R2 = {component_r2:.4f}"
            print(reprot)
            with open(LOG_FILENAME, 'a', encoding='utf-8') as f:
                f.write(reprot + "\n")
        
        t_min = self.scalers_test_min[1:, :]
        t_max = self.scalers_test_max[1:, :]

        preds_sum = np.sum(preds, axis=0) # 先只加总 IMF 分量
        preds_real = self._row_inverse_normalize(preds_sum, t_min, t_max) # 先还原到真实价格空间
        
        if self.enable_kelm:
            X_test_slice = X_test[:-1, :, :]
            X_kelm_test = X_test_slice.reshape(X_test_slice.shape[0], -1)
            residuals_pred = self.kelm_model.predict(X_kelm_test).reshape(-1, 1)
            
            preds_real = preds_real + residuals_pred

        return preds_real

    def run_multi_trend_prediction(self):
        formatted_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
        with open(LOG_FILENAME, 'a', encoding='utf-8') as f:
            f.write(f"\nstart==========> {formatted_time}\n")
            f.write(f"ticker: {self.ticker}, k: {self.K}, time_step: {self.time_step}\n")
        X_train, X_val, X_test, V_train, V_val, V_test = self._get_prepared_data(self.ticker, self.start_date, self.end_date)
        
        self._train_models(X_train, X_val, V_train, V_val)
        predicted_prices = self._predict_test_data(X_test, V_test)
        real_data = self.real_test_data[1:, :]
        self._evaluate_metrics(real_data, predicted_prices, label_name="SVMD-Local-LSTM-KELM")

    def _evaluate_metrics(self, real_data, y_pred, label_name):
        # print(f"_evaluate_metrics  real_data.shape: {real_data.shape}, y_pred.shape: {y_pred.shape}")
        # print(f"_evaluate_metrics real_data y_pred:")
        # print(real_data[:5, :])
        # print(y_pred[:5, :])
        # print("<====")
        mae = mean_absolute_error(real_data, y_pred)
        mse = mean_squared_error(real_data, y_pred)
        mape = mean_absolute_percentage_error(real_data, y_pred)
        rmse = root_mean_squared_error(real_data, y_pred)
        r2 = r2_score(real_data, y_pred)
        report_str = f"[{label_name}] 真实空间评估 -> MAE: {mae:.4f} | MAPE: {mape:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}"
        print(report_str)
        with open(LOG_FILENAME, 'a', encoding='utf-8') as f:
            f.write(report_str)

if __name__ == "__main__":
    test = True
    if test:
        ticker = "TSLA"
        k = 20
        time_step = 15
        rf_estimator = 100
        kl_flag = 2
        km_flag = 3

        predictor = VMDStrategyThreePredictor( 
                ticker=ticker, 
                start_date="2013-01-01", 
                end_date="2024-01-01",
                enable_dbo=1,
                K=k, 
                kl_flag=kl_flag,
                km_flag=km_flag,
                time_step=time_step, 
                rf_estimator=rf_estimator, 
                merged_K=3,                         
                trend_imfs=[],
                mid_imfs=[0],
                noise_imfs=[1, 2]
            )
        predictor.run_multi_trend_prediction()
    else:
        tickers = ["AAPL", "TSLA"]
        k = 20
        time_steps = [5, 10, 15, 20, 25] # 修正了同名变量问题
        rf_estimator = 100
        cnn_lstm_flag = False
        kl_flag = [1,2,3,5]
        km_flag = [1, 2, 3]

        for ticker in tickers:
            for ts in time_steps:
                for kl in kl_flag:
                    for km in km_flag:
                        predictor = VMDStrategyThreePredictor( 
                                ticker=ticker, 
                                start_date="2013-01-01", 
                                end_date="2024-01-01",
                                K=k, 
                                kl_flag=kl_flag,
                                km_flag=km_flag,
                                time_step=time_step, 
                                rf_estimator=rf_estimator, 
                                merged_K=3,                         
                                trend_imfs=[0],
                                mid_imfs=[1],
                                noise_imfs=[2]
                            )
                        predictor.run_multi_trend_prediction()