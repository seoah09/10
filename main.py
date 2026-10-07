import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="서울 연평균 기온 선형회귀 평가",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 선형회귀 모델 평가")

st.write(
    "서울의 연평균 기온 데이터를 이용하여 "
    "최근 50년과 최근 100년의 선형회귀 모델을 학습하고, "
    "공통 테스트 기간인 2006~2025년의 예측 성능을 비교합니다."
)

# =========================
# 데이터 불러오기
# =========================
URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/"
    "data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(URL, encoding="utf-8-sig")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 결측값 제거
    df = df.dropna(subset=["연도", "평균기온"])

    return df


df = load_data()

# =========================
# 연도별 평균기온 계산
# =========================
annual = (
    df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 2025년까지 사용
annual = annual[annual["연도"] <= 2025]

# 관측일수가 300일 이상인 연도만 사용
annual = annual[annual["관측일수"] >= 300]

annual = annual.sort_values("연도").reset_index(drop=True)

# =========================
# 학습 / 테스트 데이터
# =========================

# 공통 테스트 데이터
test = annual[
    (annual["연도"] >= 2006) &
    (annual["연도"] <= 2025)
].copy()

# 최근 50년 학습 데이터
train_50 = annual[
    (annual["연도"] >= 1956) &
    (annual["연도"] <= 2005)
].copy()

# 최근 100년 학습 데이터
train_100 = annual[
    (annual["연도"] >= 1906) &
    (annual["연도"] <= 2005)
].copy()


# =========================
# 선형회귀 함수
# =========================
def make_model(train_data):
    X = train_data[["연도"]]
    y = train_data["연평균기온"]

    model = LinearRegression()
    model.fit(X, y)

    return model


# =========================
# 모델 학습
# =========================
model_50 = make_model(train_50)
model_100 = make_model(train_100)


# =========================
# 테스트 예측
# =========================
X_test = test[["연도"]]
y_test = test["연평균기온"]

pred_50 = model_50.predict(X_test)
pred_100 = model_100.predict(X_test)


# =========================
# 평가 함수
# =========================
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    r2 = r2_score(y_true, y_pred)

    return mae, mse, r2


mae_50, mse_50, r2_50 = evaluate_model(
    y_test,
    pred_50
)

mae_100, mse_100, r2_100 = evaluate_model(
    y_test,
    pred_100
)


# =========================
# 전체 데이터 회귀
# =========================
model_all = make_model(annual)

pred_all = model_all.predict(
    annual[["연도"]]
)

all_r2 = r2_score(
    annual["연평균기온"],
    pred_all
)


# =========================
# 화면 출력
# =========================
st.subheader("📊 데이터 분할")

split_data = pd.DataFrame({
    "모델": [
        "최근 50년",
        "최근 100년"
    ],
    "학습 데이터": [
        "1956~2005",
        "1906~2005"
    ],
    "테스트 데이터": [
        "2006~2025",
        "2006~2025"
    ],
    "학습 연도 수": [
        len(train_50),
        len(train_100)
    ],
    "테스트 연도 수": [
        len(test),
        len(test)
    ]
})

st.dataframe(
    split_data,
    use_container_width=True,
    hide_index=True
)


# =========================
# 회귀선 기울기
# =========================
st.subheader("📈 회귀선 비교")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "최근 50년 회귀선 기울기",
        f"{model_50.coef_[0]:.4f} ℃/년"
    )

with col2:
    st.metric(
        "최근 100년 회귀선 기울기",
        f"{model_100.coef_[0]:.4f} ℃/년"
    )


# =========================
# 테스트 성능 비교
# =========================
st.subheader("🎯 테스트 데이터 예측 성능")

result = pd.DataFrame({
    "학습 기간": [
        "1956~2005 (50년)",
        "1906~2005 (100년)"
    ],
    "기울기 (℃/년)": [
        model_50.coef_[0],
        model_100.coef_[0]
    ],
    "MAE (℃)": [
        mae_50,
        mae_100
    ],
    "MSE (℃²)": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})

st.dataframe(
    result.style.format({
        "기울기 (℃/년)": "{:.4f}",
        "MAE (℃)": "{:.4f}",
        "MSE (℃²)": "{:.4f}",
        "R²": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================
# 회귀선 그래프
# =========================
st.subheader("📉 실제 기온과 회귀선 비교")

fig = go.Figure()

# 실제 연평균 기온
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균 기온"
    )
)

# 50년 회귀선
x_line_50 = np.arange(1956, 2026)

y_line_50 = model_50.predict(
    x_line_50.reshape(-1, 1)
)

fig.add_trace(
    go.Scatter(
        x=x_line_50,
        y=y_line_50,
        mode="lines",
        name="50년 회귀선"
    )
)

# 100년 회귀선
x_line_100 = np.arange(1906, 2026)

y_line_100 = model_100.predict(
    x_line_100.reshape(-1, 1)
)

fig.add_trace(
    go.Scatter(
        x=x_line_100,
        y=y_line_100,
        mode="lines",
        name="100년 회귀선"
    )
)

# 테스트 기간 표시
fig.add_vrect(
    x0=2006,
    x1=2025,
    fillcolor="gray",
    opacity=0.15,
    line_width=0,
    annotation_text="공통 테스트 기간",
    annotation_position="top left"
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================
# 전체 데이터 적합도
# =========================
st.subheader("📌 전체 데이터에 대한 평가")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "전체 데이터 연도 수",
        f"{len(annual)}년"
    )

with col2:
    st.metric(
        "전체 데이터 회귀선 기울기",
        f"{model_all.coef_[0]:.4f} ℃/년"
    )

with col3:
    st.metric(
        "전체 데이터 R²",
        f"{all_r2:.4f}"
    )


# =========================
# 해석
# =========================
st.subheader("📝 결과 해석")

if mae_50 < mae_100:
    better_mae = "최근 50년 모델"
else:
    better_mae = "최근 100년 모델"

if mse_50 < mse_100:
    better_mse = "최근 50년 모델"
else:
    better_mse = "최근 100년 모델"

if r2_50 > r2_100:
    better_r2 = "최근 50년 모델"
else:
    better_r2 = "최근 100년 모델"

st.write(
    f"""
- **MAE 기준:** {better_mae}의 평균 절대 오차가 더 작습니다.
- **MSE 기준:** {better_mse}의 제곱 오차가 더 작습니다.
- **R² 기준:** {better_r2}의 테스트 데이터 설명력이 더 높습니다.
- 두 모델 모두 **2006~2025년이라는 동일한 테스트 데이터**를 사용했기 때문에
  두 학습 기간의 예측 성능을 공정하게 비교할 수 있습니다.
"""
)

st.info(
    "MAE와 MSE는 작을수록 예측 성능이 좋고, "
    "R²는 클수록 실제 기온 변화를 잘 설명합니다."
)
