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

# 공휴일 로드 및 세트화
holidays_df = pd.read_csv('./holiday.csv')
holiday_dates = set(pd.to_datetime(holidays_df['date']).dt.date)

# 날짜 기반 피처 정의
weekday_names = ['월', '화', '수', '목', '금', '토', '일']
weekday_cols = [f'요일_{w}' for w in weekday_names]
month_cols = [f'월_{m}' for m in range(1, 13)]
season_names = ['봄', '여름', '가을', '겨울']
season_cols = [f'계절_{s}' for s in season_names]

FEATURE_COLS = ['매출수량'] + weekday_cols + ['공휴일'] + month_cols + season_cols

def month_to_season(m: int) -> str:
    # 사용자 정의: 봄(4,5) / 여름(6,7,8,9) / 가을(10,11) / 겨울(12,1,2,3)
    if m in (4, 5):
        return '봄'
    if m in (6, 7, 8, 9):
        return '여름'
    if m in (10, 11):
        return '가을'
    return '겨울'  # 12, 1, 2, 3

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df['영업일자'])
    # 요일 원-핫 (월=0)
    dow = dt.dt.weekday
    for i, w in enumerate(weekday_names):
        df[f'요일_{w}'] = (dow == i).astype(int)
    # 공휴일 플래그
    df['공휴일'] = dt.dt.date.map(lambda d: 1 if d in holiday_dates else 0).astype(int)
    # 월 원-핫
    mon = dt.dt.month
    for m in range(1, 13):
        df[f'월_{m}'] = (mon == m).astype(int)
    # 계절 원-핫
    seasons = mon.map(month_to_season)
    for s in season_names:
        df[f'계절_{s}'] = (seasons == s).astype(int)
    return df

# 영업장/메뉴 파생 컬럼 생성 유틸
store_names = [
    '느티나무', '담하', '라그로타', '미라시아', '연회장',
    '카페테리아', '포레스트릿', '화담숲주막', '화담숲카페'
]

def _normalize_text(value: object) -> str:
    s = str(value) if not isinstance(value, str) else value
    # 특수 공백 제거/치환
    s = s.replace('\u200b', '').replace('\xa0', ' ')
    s = s.strip()
    # 다중 공백 축약
    s = re.sub(r"\s+", " ", s)
    return s

def split_store_menu_by_list(x, store_names):
    s = _normalize_text(x)
    # 1) 가장 신뢰도 높은 규칙: 첫 '_' 기준 분리
    if '_' in s:
        store, menu = s.split('_', 1)
        return store.strip(), menu.strip()
    # 2) 백업: 사전 정의된 영업장 접두어 매칭
    for store in store_names:
        if s.startswith(store):
            return store, s[len(store):].lstrip(' _')
    # 3) 실패 시 원본 유지, 메뉴는 공백
    return s, ''

# 원본 `영업장명_메뉴명`은 그대로 유지하고 파생 컬럼만 추가
train[['영업장', '메뉴']] = train['영업장명_메뉴명'].apply(
    lambda x: pd.Series(split_store_menu_by_list(x, store_names))
)

# 분리 결과 상위 10행 출력
print("\n[영업장/메뉴 분리 결과 상위 10행]")
try:
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(train[['영업일자', '영업장명_메뉴명', '영업장', '메뉴']].head(10).to_string(index=False))
except Exception:
    print(train[['영업장명_메뉴명', '영업장', '메뉴']].head(10))

# 전처리 완료(train) 상위 10행 출력
preprocessed_train = add_calendar_features(train)
print("\n[전처리 완료 train 상위 10행]")
try:
    cols_to_show = ['영업일자', '영업장명_메뉴명', '영업장', '메뉴', '매출수량'] + weekday_cols + ['공휴일'] + month_cols + season_cols
    cols_to_show = [c for c in cols_to_show if c in preprocessed_train.columns]
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(preprocessed_train[cols_to_show].head(10).to_string(index=False))
except Exception:
    print(preprocessed_train.head(10))

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

    for key, group in tqdm(train_df.groupby(['영업장명_메뉴명']), desc='Training LSTM'):
        store_train = group.sort_values('영업일자').copy()
        if len(store_train) < LOOKBACK + PREDICT:
            continue

        # 캘린더 피처 추가
        store_train = add_calendar_features(store_train)

        # 스케일러 분리: 입력(feature)과 타깃(매출수량)
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        feature_matrix = store_train[FEATURE_COLS].values
        feature_matrix_scaled = feature_scaler.fit_transform(feature_matrix)

        target_array = store_train[['매출수량']].values
        target_array_scaled = target_scaler.fit_transform(target_array).reshape(-1)

        # 시퀀스 구성
        X_train, y_train = [], []
        for i in range(len(feature_matrix_scaled) - LOOKBACK - PREDICT + 1):
            X_train.append(feature_matrix_scaled[i:i+LOOKBACK])
            y_train.append(target_array_scaled[i+LOOKBACK:i+LOOKBACK+PREDICT])

        X_train = torch.tensor(np.array(X_train)).float().to(DEVICE)
        y_train = torch.tensor(np.array(y_train)).float().to(DEVICE)

        model = MultiOutputLSTM(input_dim=len(FEATURE_COLS), hidden_dim=256, num_layers=3, output_dim=PREDICT, dropout=0.3).to(DEVICE)
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
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler
        }

    return trained_models

#%%
# 학습
trained_models = train_model(train)

#%%
def predict_model(test_df, trained_models, test_prefix: str):
    results = []
    debug_count = 0

    for key, store_test in test_df.groupby(['영업장명_메뉴명']):
        if key not in trained_models:
            continue

        model = trained_models[key]['model']
        feature_scaler = trained_models[key]['feature_scaler']
        target_scaler = trained_models[key]['target_scaler']

        store_test_sorted = store_test.sort_values('영업일자')
        store_test_sorted = add_calendar_features(store_test_sorted)

        if len(store_test_sorted) < LOOKBACK:
            continue

        # 입력 스케일링
        recent_features = store_test_sorted[FEATURE_COLS].values[-LOOKBACK:]
        recent_features_scaled = feature_scaler.transform(recent_features)
        x_input = torch.tensor([recent_features_scaled]).float().to(DEVICE)

        with torch.no_grad():
            pred_scaled = model(x_input).squeeze().cpu().numpy()

        # 역변환
        restored = []
        for i in range(PREDICT):
            dummy_df = pd.DataFrame([[pred_scaled[i]]], columns=["매출수량"])
            restored_val = target_scaler.inverse_transform(dummy_df)[0, 0]
            restored.append(max(restored_val, 0))

        if debug_count < 2:
            print(f"[디버그] key={key}")
            print(f"  recent_features.shape={recent_features.shape}")
            print(f"  recent_features_scaled.shape={np.array(recent_features_scaled).shape}")
            print(f"  pred_scaled={pred_scaled}")
            print(f"  restored={restored}")
            debug_count += 1

        # 예측일자: TEST_00+1일 ~ TEST_00+7일
        pred_dates = [f"{test_prefix}+{i+1}일" for i in range(PREDICT)]

        for d, val in zip(pred_dates, restored):
            results.append({
                '영업일자': d,
                '영업장명_메뉴명': key,
                '매출수량': val
            })

    pred_df = pd.DataFrame(results)
    return pred_df

#%%
all_preds = []

# 모든 test_*.csv 순회
test_files = sorted(glob.glob('./test/TEST_*.csv'))

for path in test_files:
    test_df = pd.read_csv(path)

    # 원본을 유지하면서 파생 컬럼 추가
    test_df[['영업장', '메뉴']] = test_df['영업장명_메뉴명'].apply(
        lambda x: pd.Series(split_store_menu_by_list(x, store_names))
    )

    # 파일명에서 접두어 추출 (예: TEST_00)
    filename = os.path.basename(path)
    test_prefix = re.search(r'(TEST_\d+)', filename).group(1)

    pred_df = predict_model(test_df, trained_models, test_prefix)
    all_preds.append(pred_df)
    
full_pred_df = pd.concat(all_preds, ignore_index=True)

#%%
def convert_to_submission_format(pred_df: pd.DataFrame, sample_submission: pd.DataFrame):
    # (영업일자, 영업장명_메뉴명) → 매출수량 딕셔너리로 변환
    pred_dict = dict(zip(
        zip(pred_df['영업일자'], pred_df['영업장명_메뉴명']),
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
