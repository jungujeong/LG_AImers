#%%
import os
import random
import glob
import re

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from tqdm import tqdm

#%%
def print_gpu_status():
    print("[GPU 사용 가능 여부]", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print(f"[사용중인 GPU 개수] {torch.cuda.device_count()}")
            current = torch.cuda.current_device()
            print(f"[현재 선택된 GPU] {current} : {torch.cuda.get_device_name(current)}")
            print(f"[GPU 메모리 사용량] {torch.cuda.memory_allocated() / 1024**2:.2f} MB / {torch.cuda.memory_reserved() / 1024**2:.2f} MB (allocated / reserved)")
        except Exception as e:
            print(f"GPU 상태 조회 중 예외 발생: {e}")
    else:
        print("GPU를 사용할 수 없습니다. CPU만 사용 중입니다.")

print_gpu_status()

#%%
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

#%%
LOOKBACK, PREDICT, BATCH_SIZE, EPOCHS = 28, 7, 256, 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
train = pd.read_csv('./train/train.csv')

# 영업장/메뉴 분리 유틸
def split_store_menu_by_list(x, store_names):
    for store in store_names:
        if isinstance(x, str) and x.startswith(store):
            return store, x[len(store):].lstrip(' _')
    return x, ''

store_names = [
    '느티나무', '담하', '라그로타', '미라시아', '연회장',
    '카페테리아', '포레스트릿', '화담숲주막', '화담숲카페'
]

# 영업장/메뉴 컬럼 생성
train[['영업장', '메뉴']] = train['영업장명_메뉴명'].apply(lambda x: pd.Series(split_store_menu_by_list(x, store_names)))

# (옵션) 요일 컬럼 생성
def add_weekday_columns(df, date_col='영업일자'):
    df[date_col] = pd.to_datetime(df[date_col])
    weekdays = ['월', '화', '수', '목', '금', '토', '일']
    for i, w in enumerate(weekdays):
        df[f'요일_{w}'] = (df[date_col].dt.weekday == i).astype(int)
    return df

train = add_weekday_columns(train)

#%%
class MultiOutputLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, num_layers=3, output_dim=7, dropout=0.3):
        super(MultiOutputLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # (B, output_dim)

#%%
def train_model(train_df):
    trained_models = {}
    patience = 10

    for key, group in tqdm(train_df.groupby(['영업장', '메뉴']), desc='Training LSTM'):
        store_train = group.sort_values('영업일자').copy()
        if len(store_train) < LOOKBACK + PREDICT:
            continue

        features = ['매출수량']
        scaler = MinMaxScaler()
        store_train[features] = scaler.fit_transform(store_train[features])
        train_vals = store_train[features].values  # shape: (N, 1)

        # 시퀀스 구성
        X_train, y_train = [], []
        for i in range(len(train_vals) - LOOKBACK - PREDICT + 1):
            X_train.append(train_vals[i:i+LOOKBACK])
            y_train.append(train_vals[i+LOOKBACK:i+LOOKBACK+PREDICT, 0])

        X_train = torch.tensor(X_train).float().to(DEVICE)
        y_train = torch.tensor(y_train).float().to(DEVICE)

        model = MultiOutputLSTM(input_dim=1, hidden_dim=256, num_layers=3, output_dim=PREDICT, dropout=0.3).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        criterion = nn.HuberLoss()

        best_loss = float('inf')
        best_state = None
        patience_counter = 0

        model.train()
        for epoch in range(EPOCHS):
            idx = torch.randperm(len(X_train))
            epoch_loss = 0.0
            for i in range(0, len(X_train), BATCH_SIZE):
                batch_idx = idx[i:i+BATCH_SIZE]
                X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
                output = model(X_batch)
                loss = criterion(output, y_batch)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)

            epoch_loss /= max(len(X_train), 1)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"[Early Stopping] {key} epoch {epoch+1} loss {epoch_loss:.5f}")
                break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"[LSTM][{key}] epoch {epoch+1} loss {epoch_loss:.5f}")

        if best_state is not None:
            model.load_state_dict(best_state)

        trained_models[key] = {
            'model': model.eval(),
            'scaler': scaler,
            'last_sequence': train_vals[-LOOKBACK:]  # (28, 1)
        }

    return trained_models

#%%
# 학습
trained_models = train_model(train)

#%%
def predict_model(test_df, trained_models, test_prefix: str):
    results = []
    debug_count = 0

    for key, store_test in test_df.groupby(['영업장', '메뉴']):
        if key not in trained_models:
            continue

        model = trained_models[key]['model']
        scaler = trained_models[key]['scaler']

        store_test_sorted = store_test.sort_values('영업일자')
        recent_vals = store_test_sorted['매출수량'].values[-LOOKBACK:]
        if len(recent_vals) < LOOKBACK:
            continue

        # 정규화(DataFrame 사용으로 스케일러 안전성 강화)
        recent_vals_df = pd.DataFrame(recent_vals.reshape(-1, 1), columns=["매출수량"])
        recent_vals_scaled = scaler.transform(recent_vals_df)
        x_input = torch.tensor([recent_vals_scaled]).float().to(DEVICE)

        with torch.no_grad():
            pred_scaled = model(x_input).squeeze().cpu().numpy()

        # 역변환
        restored = []
        for i in range(PREDICT):
            dummy = np.zeros((1, 1))
            dummy[0, 0] = pred_scaled[i]
            dummy_df = pd.DataFrame(dummy, columns=["매출수량"])
            restored_val = scaler.inverse_transform(dummy_df)[0, 0]
            restored.append(max(restored_val, 0))

        if debug_count < 2:
            print(f"[디버그] key={key}")
            print(f"  recent_vals={recent_vals}")
            print(f"  recent_vals_scaled={np.array(recent_vals_scaled).flatten()}")
            print(f"  pred_scaled={pred_scaled}")
            print(f"  restored={restored}")
            debug_count += 1

        # 예측일자: TEST_00+1일 ~ TEST_00+7일
        pred_dates = [f"{test_prefix}+{i+1}일" for i in range(PREDICT)]

        for d, val in zip(pred_dates, restored):
            results.append({
                '영업일자': d,
                '영업장': key[0],
                '메뉴': key[1],
                '매출수량': val
            })

    pred_df = pd.DataFrame(results)
    if not pred_df.empty:
        pred_df['업장_메뉴'] = pred_df['영업장'] + ' ' + pred_df['메뉴']
    return pred_df

#%%
all_preds = []

# 모든 test_*.csv 순회
test_files = sorted(glob.glob('./test/TEST_*.csv'))

for path in test_files:
    test_df = pd.read_csv(path)
    # 영업장/메뉴 분리
    test_df[['영업장', '메뉴']] = test_df['영업장명_메뉴명'].apply(lambda x: pd.Series(split_store_menu_by_list(x, store_names)))

    # 파일명에서 접두어 추출 (예: TEST_00)
    filename = os.path.basename(path)
    test_prefix = re.search(r'(TEST_\d+)', filename).group(1)

    pred_df = predict_model(test_df, trained_models, test_prefix)
    all_preds.append(pred_df)
    
full_pred_df = pd.concat(all_preds, ignore_index=True)

#%%
def convert_to_submission_format(pred_df: pd.DataFrame, sample_submission: pd.DataFrame):
    # (영업일자, 업장_메뉴) → 매출수량 딕셔너리로 변환
    pred_dict = dict(zip(
        zip(pred_df['영업일자'], pred_df['업장_메뉴']),
        pred_df['매출수량']
    ))

    final_df = sample_submission.copy()

    for row_idx in final_df.index:
        date = final_df.loc[row_idx, '영업일자']
        for col in final_df.columns[1:]:  # 메뉴명들
            # 사용자가 검증한 고득점 매핑을 유지
            final_df.loc[row_idx, col] = pred_dict.get((date, (col,)), 0)

    return final_df

#%%
sample_submission = pd.read_csv('./sample_submission.csv')
submission = convert_to_submission_format(full_pred_df, sample_submission)
submission.to_csv('baseline_submission.csv', index=False, encoding='utf-8-sig')
# %%
