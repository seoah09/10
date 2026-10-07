import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="서울 연평균 기온 선형회귀 분석",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 선형회귀 분석")

st.write(
    "서울의 연평균 기온 데이터를 이용하여 "
    "50년 학습 모델과 100년 학습 모델을 만들고 "
    "최근 20년(2006~2025)의 실제 기온으로 예측 성능을 비교합니다."
)

# =========================================================
# 데이터 불러오기
# =========================================================
URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/"
    "data/seoul.csv"
)


@st.cache_data
def load_data():

    df = pd.read_csv(
        URL,
        encoding="utf-8-sig"
    )

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 결측값 제거
    df = df.dropna(
        subset=["날짜", "평균기온"]
    )

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()

# =========================================================
# 연평균 기온 계산
# =========================================================
annual = (
    df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 2025년까지 사용
annual = annual[
    annual["연도"] <= 2025
]

# 관측일수가 300일 이상인 연도만 사용
annual = annual[
    annual["관측일수"] >= 300
]

annual = annual.sort_values(
    "연도"
).reset_index(drop=True)


# =========================================================
# 학습 / 테스트 데이터
# =========================================================

# 최근 50년 학습
train_50 = annual[
    (annual["연도"] >= 1956) &
    (annual["연도"] <= 2005)
].copy()

# 최근 100년 학습
train_100 = annual[
    (annual["연도"] >= 1906) &
    (annual["연도"] <= 2005)
].copy()

# 공통 테스트 데이터
test = annual[
    (annual["연도"] >= 2006) &
    (annual["연도"] <= 2025)
].copy()


# =========================================================
# 선형회귀 모델
# =========================================================

def create_model(train_data):

    X = train_data[["연도"]]
    y = train_data["연평균기온"]

    model = LinearRegression()

    model.fit(X, y)

    return model


model_50 = create_model(train_50)

model_100 = create_model(train_100)


# =========================================================
# 최근 20년 예측
# =========================================================

X_test = test[["연도"]]

y_test = test["연평균기온"]


test["50년_예측"] = model_50.predict(
    X_test
)

test["100년_예측"] = model_100.predict(
    X_test
)


# =========================================================
# 예측 오차 계산
# =========================================================

test["50년_오차"] = (
    test["50년_예측"] -
    test["연평균기온"]
)

test["100년_오차"] = (
    test["100년_예측"] -
    test["연평균기온"]
)

test["50년_절대오차"] = (
    test["50년_오차"].abs()
)

test["100년_절대오차"] = (
    test["100년_오차"].abs()
)


# =========================================================
# 모델 평가
# =========================================================

mae_50 = mean_absolute_error(
    y_test,
    test["50년_예측"]
)

mse_50 = mean_squared_error(
    y_test,
    test["50년_예측"]
)

r2_50 = r2_score(
    y_test,
    test["50년_예측"]
)


mae_100 = mean_absolute_error(
    y_test,
    test["100년_예측"]
)

mse_100 = mean_squared_error(
    y_test,
    test["100년_예측"]
)

r2_100 = r2_score(
    y_test,
    test["100년_예측"]
)


# =========================================================
# 제목
# =========================================================
st.header("1. 학습 데이터와 테스트 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "50년 학습 데이터",
        f"{len(train_50)}년",
        "1956~2005"
    )

with col2:
    st.metric(
        "100년 학습 데이터",
        f"{len(train_100)}년",
        "1906~2005"
    )

with col3:
    st.metric(
        "공통 테스트 데이터",
        f"{len(test)}년",
        "2006~2025"
    )


st.info(
    "두 모델 모두 2006~2025년의 동일한 데이터를 테스트 데이터로 사용합니다."
)


# =========================================================
# 모델 회귀식
# =========================================================
st.header("2. 두 회귀 모델의 회귀식")

col1, col2 = st.columns(2)

with col1:

    st.subheader("📘 최근 50년 학습 모델")

    st.write(
        f"기울기: **{model_50.coef_[0]:.5f} ℃/년**"
    )

    st.write(
        f"절편: **{model_50.intercept_:.3f}**"
    )

    st.code(
        f"예측 기온 = "
        f"{model_50.coef_[0]:.5f} × 연도 "
        f"+ {model_50.intercept_:.3f}"
    )


with col2:

    st.subheader("📗 최근 100년 학습 모델")

    st.write(
        f"기울기: **{model_100.coef_[0]:.5f} ℃/년**"
    )

    st.write(
        f"절편: **{model_100.intercept_:.3f}**"
    )

    st.code(
        f"예측 기온 = "
        f"{model_100.coef_[0]:.5f} × 연도 "
        f"+ {model_100.intercept_:.3f}"
    )


# =========================================================
# 테스트 성능
# =========================================================
st.header("3. 최근 20년 테스트 데이터 예측 성능")

st.subheader("📊 50년 학습 모델")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE",
        f"{mae_50:.4f} ℃"
    )

with col2:
    st.metric(
        "MSE",
        f"{mse_50:.4f} ℃²"
    )

with col3:
    st.metric(
        "R²",
        f"{r2_50:.4f}"
    )


st.subheader("📊 100년 학습 모델")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE",
        f"{mae_100:.4f} ℃"
    )

with col2:
    st.metric(
        "MSE",
        f"{mse_100:.4f} ℃²"
    )

with col3:
    st.metric(
        "R²",
        f"{r2_100:.4f}"
    )


# =========================================================
# 성능 비교 표
# =========================================================
st.subheader("📋 두 모델 성능 비교")

performance = pd.DataFrame({
    "모델": [
        "50년 학습",
        "100년 학습"
    ],
    "학습기간": [
        "1956~2005",
        "1906~2005"
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
    performance.style.format({
        "기울기 (℃/년)": "{:.5f}",
        "MAE (℃)": "{:.4f}",
        "MSE (℃²)": "{:.4f}",
        "R²": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 최근 20년 실제 vs 예측
# =========================================================
st.header("4. 최근 20년 실제 기온과 예측 기온")

st.write(
    "2006~2025년 각각에 대해 실제 연평균 기온과 "
    "50년 학습 모델, 100년 학습 모델의 예측값을 비교합니다."
)


# 보기 좋은 표
comparison = test[
    [
        "연도",
        "연평균기온",
        "50년_예측",
        "100년_예측",
        "50년_오차",
        "100년_오차"
    ]
].copy()


comparison.columns = [
    "연도",
    "실제 기온 (℃)",
    "50년 예측 (℃)",
    "100년 예측 (℃)",
    "50년 오차 (℃)",
    "100년 오차 (℃)"
]


st.dataframe(
    comparison.style.format({
        "실제 기온 (℃)": "{:.2f}",
        "50년 예측 (℃)": "{:.2f}",
        "100년 예측 (℃)": "{:.2f}",
        "50년 오차 (℃)": "{:.2f}",
        "100년 오차 (℃)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 실제 기온 vs 예측 기온 그래프
# =========================================================
st.subheader("📈 실제 기온과 두 모델의 예측 기온")

fig = go.Figure()


# 실제 기온
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="lines+markers",
        name="실제 기온"
    )
)


# 50년 예측
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["50년_예측"],
        mode="lines+markers",
        name="50년 학습 모델"
    )
)


# 100년 예측
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["100년_예측"],
        mode="lines+markers",
        name="100년 학습 모델"
    )
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


# =========================================================
# 연도별 절대오차
# =========================================================
st.header("5. 연도별 예측 오차 비교")

error_table = test[
    [
        "연도",
        "50년_절대오차",
        "100년_절대오차"
    ]
].copy()

error_table.columns = [
    "연도",
    "50년 모델 절대오차 (℃)",
    "100년 모델 절대오차 (℃)"
]


st.dataframe(
    error_table.style.format({
        "50년 모델 절대오차 (℃)": "{:.2f}",
        "100년 모델 절대오차 (℃)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 오차 그래프
# =========================================================
st.subheader("📉 연도별 절대오차")

error_fig = go.Figure()


error_fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["50년_절대오차"],
        mode="lines+markers",
        name="50년 학습 모델"
    )
)


error_fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["100년_절대오차"],
        mode="lines+markers",
        name="100년 학습 모델"
    )
)


error_fig.update_layout(
    xaxis_title="연도",
    yaxis_title="절대오차 (℃)",
    hovermode="x unified"
)


st.plotly_chart(
    error_fig,
    use_container_width=True
)


# =========================================================
# 어떤 모델이 더 좋은지 숫자로 비교
# =========================================================
st.header("6. 테스트 성능 비교 결과")

mae_difference = abs(mae_50 - mae_100)
mse_difference = abs(mse_50 - mse_100)
r2_difference = abs(r2_50 - r2_100)


col1, col2 = st.columns(2)


with col1:

    st.subheader("🏆 MAE")

    if mae_50 < mae_100:

        st.success(
            f"50년 학습 모델이 더 좋습니다. "
            f"(MAE 차이: {mae_difference:.4f}℃)"
        )

    elif mae_100 < mae_50:

        st.success(
            f"100년 학습 모델이 더 좋습니다. "
            f"(MAE 차이: {mae_difference:.4f}℃)"
        )

    else:

        st.info(
            "두 모델의 MAE가 같습니다."
        )


with col2:

    st.subheader("🏆 R²")

    if r2_50 > r2_100:

        st.success(
            f"50년 학습 모델이 더 좋습니다. "
            f"(R² 차이: {r2_difference:.4f})"
        )

    elif r2_100 > r2_50:

        st.success(
            f"100년 학습 모델이 더 좋습니다. "
            f"(R² 차이: {r2_difference:.4f})"
        )

    else:

        st.info(
            "두 모델의 R²가 같습니다."
        )


# =========================================================
# 전체 데이터 회귀
# =========================================================
st.header("7. 전체 연평균 기온 데이터의 선형회귀")

model_all = create_model(annual)

all_prediction = model_all.predict(
    annual[["연도"]]
)

all_mae = mean_absolute_error(
    annual["연평균기온"],
    all_prediction
)

all_mse = mean_squared_error(
    annual["연평균기온"],
    all_prediction
)

all_r2 = r2_score(
    annual["연평균기온"],
    all_prediction
)


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "사용 연도",
        f"{len(annual)}년"
    )

with col2:
    st.metric(
        "기울기",
        f"{model_all.coef_[0]:.5f} ℃/년"
    )

with col3:
    st.metric(
        "MAE",
        f"{all_mae:.4f} ℃"
    )

with col4:
    st.metric(
        "R²",
        f"{all_r2:.4f}"
    )


# =========================================================
# 데이터 확인
# =========================================================
with st.expander("📂 분석에 사용된 연평균 기온 데이터 보기"):

    st.dataframe(
        annual,
        use_container_width=True,
        hide_index=True
    )
