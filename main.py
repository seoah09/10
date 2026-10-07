import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연평균 기온 데이터를 이용하여 선형회귀 분석과 "
    "50년·100년 학습모델의 예측 성능을 비교합니다."
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
        encoding="utf-8"
    )

    # 날짜를 날짜 형식으로 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온을 숫자로 변환
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
# 연도별 평균기온 계산
# =========================================================

annual = (
    df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 2025년 이후 데이터 제외
annual = annual[
    annual["연도"] <= 2025
]

# 관측일수가 300일 미만인 해 제외
annual = annual[
    annual["관측일수"] >= 300
]

annual = annual.sort_values(
    "연도"
).reset_index(drop=True)


# =========================================================
# 전체 데이터 선형회귀
# =========================================================

# 1908년부터 지난 연수를 독립변수로 사용
annual["경과연수"] = annual["연도"] - 1908

X_all = annual[["경과연수"]]
y_all = annual["연평균기온"]

model_all = LinearRegression()
model_all.fit(X_all, y_all)

annual["회귀예측"] = model_all.predict(X_all)

# 상관계수
correlation = np.corrcoef(
    annual["경과연수"],
    annual["연평균기온"]
)[0, 1]

# 전체 데이터 평가
all_mae = mean_absolute_error(
    y_all,
    annual["회귀예측"]
)

all_mse = mean_squared_error(
    y_all,
    annual["회귀예측"]
)

all_r2 = r2_score(
    y_all,
    annual["회귀예측"]
)


# =========================================================
# 기본 데이터 정보
# =========================================================

st.header("1. 분석에 사용된 데이터")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "사용 연도 수",
        f"{len(annual)}년"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{annual['연도'].min()}년"
    )

with col3:
    st.metric(
        "끝 연도",
        f"{annual['연도'].max()}년"
    )

with col4:
    st.metric(
        "상관계수",
        f"{correlation:.4f}"
    )


st.info(
    "2025년까지의 데이터 중 연간 관측일수가 300일 이상인 연도만 사용했습니다."
)


# =========================================================
# 전체 데이터 산점도 + 회귀선
# =========================================================

st.header("2. 전체 연평균 기온과 선형회귀")

st.write(
    "독립변수는 1908년부터의 경과연수이며, "
    "그래프의 가로축은 실제 연도를 표시합니다."
)

fig_all = go.Figure()


# 실제 연평균 기온
fig_all.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균 기온"
    )
)


# 회귀선
fig_all.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["회귀예측"],
        mode="lines",
        name="선형회귀선"
    )
)


fig_all.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)


st.plotly_chart(
    fig_all,
    use_container_width=True
)


# =========================================================
# 전체 데이터 회귀 결과
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "전체 데이터 MAE",
        f"{all_mae:.4f} ℃"
    )

with col2:
    st.metric(
        "전체 데이터 MSE",
        f"{all_mse:.4f} ℃²"
    )

with col3:
    st.metric(
        "전체 데이터 R²",
        f"{all_r2:.4f}"
    )


st.write(
    f"**전체 회귀선 기울기:** "
    f"{model_all.coef_[0]:.5f} ℃/년"
)


# =========================================================
# 연도 슬라이더
# =========================================================

st.header("3. 연도별 예상 기온")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

selected_elapsed = selected_year - 1908

predicted_temp = model_all.predict(
    np.array([[selected_elapsed]])
)[0]


st.metric(
    f"{selected_year}년 예상 연평균 기온",
    f"{predicted_temp:.2f} ℃"
)


# =========================================================
# 학습 / 테스트 데이터
# =========================================================

st.header("4. 학습 데이터와 테스트 데이터")

# 50년 학습
train_50 = annual[
    (annual["연도"] >= 1956) &
    (annual["연도"] <= 2005)
].copy()

# 100년 학습
train_100 = annual[
    (annual["연도"] >= 1906) &
    (annual["연도"] <= 2005)
].copy()

# 공통 테스트 데이터
test = annual[
    (annual["연도"] >= 2006) &
    (annual["연도"] <= 2025)
].copy()


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "50년 학습",
        f"{len(train_50)}년"
    )
    st.caption("1956~2005")

with col2:
    st.metric(
        "100년 학습",
        f"{len(train_100)}년"
    )
    st.caption("1906~2005")

with col3:
    st.metric(
        "공통 테스트",
        f"{len(test)}년"
    )
    st.caption("2006~2025")


# =========================================================
# 50년 / 100년 모델 생성
# =========================================================

def create_regression_model(train_data):

    X = train_data[["경과연수"]]
    y = train_data["연평균기온"]

    model = LinearRegression()

    model.fit(X, y)

    return model


model_50 = create_regression_model(
    train_50
)

model_100 = create_regression_model(
    train_100
)


# =========================================================
# 테스트 데이터 예측
# =========================================================

X_test = test[["경과연수"]]
y_test = test["연평균기온"]


test["50년_예측기온"] = model_50.predict(
    X_test
)

test["100년_예측기온"] = model_100.predict(
    X_test
)


# =========================================================
# 테스트 성능 평가
# =========================================================

mae_50 = mean_absolute_error(
    y_test,
    test["50년_예측기온"]
)

mse_50 = mean_squared_error(
    y_test,
    test["50년_예측기온"]
)

r2_50 = r2_score(
    y_test,
    test["50년_예측기온"]
)


mae_100 = mean_absolute_error(
    y_test,
    test["100년_예측기온"]
)

mse_100 = mean_squared_error(
    y_test,
    test["100년_예측기온"]
)

r2_100 = r2_score(
    y_test,
    test["100년_예측기온"]
)


# =========================================================
# 50년 / 100년 회귀선 기울기
# =========================================================

slope_50 = model_50.coef_[0]
slope_100 = model_100.coef_[0]


# =========================================================
# 모델 성능 카드
# =========================================================

st.header("5. 50년 학습 모델 테스트 성능")

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


st.write(
    f"**회귀선 기울기:** {slope_50:.5f} ℃/년"
)


st.header("6. 100년 학습 모델 테스트 성능")

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


st.write(
    f"**회귀선 기울기:** {slope_100:.5f} ℃/년"
)


# =========================================================
# 성능 비교 표
# =========================================================

st.header("7. 50년 학습 모델과 100년 학습 모델 비교")

performance = pd.DataFrame({
    "모델": [
        "50년 학습 모델",
        "100년 학습 모델"
    ],
    "학습 기간": [
        "1956~2005",
        "1906~2005"
    ],
    "테스트 기간": [
        "2006~2025",
        "2006~2025"
    ],
    "기울기 (℃/년)": [
        slope_50,
        slope_100
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

st.header("8. 최근 20년 실제 기온과 예측 기온")

st.write(
    "2006~2025년의 실제 연평균 기온과 "
    "50년·100년 학습 모델의 예측값을 연도별로 비교합니다."
)


comparison = test[
    [
        "연도",
        "연평균기온",
        "50년_예측기온",
        "100년_예측기온"
    ]
].copy()


comparison.columns = [
    "연도",
    "실제 기온 (℃)",
    "50년 예측 (℃)",
    "100년 예측 (℃)"
]


st.dataframe(
    comparison.style.format({
        "실제 기온 (℃)": "{:.2f}",
        "50년 예측 (℃)": "{:.2f}",
        "100년 예측 (℃)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 실제 vs 예측 그래프
# =========================================================

st.subheader("📈 실제 기온과 예측 기온 비교")

fig_test = go.Figure()


fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="lines+markers",
        name="실제 기온"
    )
)


fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["50년_예측기온"],
        mode="lines+markers",
        name="50년 학습 모델"
    )
)


fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["100년_예측기온"],
        mode="lines+markers",
        name="100년 학습 모델"
    )
)


fig_test.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)


st.plotly_chart(
    fig_test,
    use_container_width=True
)


# =========================================================
# 연도별 예측 오차
# =========================================================

st.header("9. 최근 20년 연도별 예측 오차")

test["50년_오차"] = (
    test["50년_예측기온"] -
    test["연평균기온"]
)

test["100년_오차"] = (
    test["100년_예측기온"] -
    test["연평균기온"]
)

test["50년_절대오차"] = (
    test["50년_오차"].abs()
)

test["100년_절대오차"] = (
    test["100년_오차"].abs()
)


error_table = test[
    [
        "연도",
        "50년_오차",
        "100년_오차",
        "50년_절대오차",
        "100년_절대오차"
    ]
].copy()


error_table.columns = [
    "연도",
    "50년 오차 (℃)",
    "100년 오차 (℃)",
    "50년 절대오차 (℃)",
    "100년 절대오차 (℃)"
]


st.dataframe(
    error_table.style.format({
        "50년 오차 (℃)": "{:.2f}",
        "100년 오차 (℃)": "{:.2f}",
        "50년 절대오차 (℃)": "{:.2f}",
        "100년 절대오차 (℃)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 절대오차 그래프
# =========================================================

st.subheader("📉 모델별 연도별 절대오차")

fig_error = go.Figure()


fig_error.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["50년_절대오차"],
        mode="lines+markers",
        name="50년 학습 모델"
    )
)


fig_error.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["100년_절대오차"],
        mode="lines+markers",
        name="100년 학습 모델"
    )
)


fig_error.update_layout(
    xaxis_title="연도",
    yaxis_title="절대오차 (℃)",
    hovermode="x unified"
)


st.plotly_chart(
    fig_error,
    use_container_width=True
)


# =========================================================
# 모델 비교 결과
# =========================================================

st.header("10. 모델 비교 결과")

comparison_result = pd.DataFrame({
    "평가지표": [
        "MAE",
        "MSE",
        "R²"
    ],
    "50년 학습": [
        mae_50,
        mse_50,
        r2_50
    ],
    "100년 학습": [
        mae_100,
        mse_100,
        r2_100
    ]
})


st.dataframe(
    comparison_result.style.format({
        "50년 학습": "{:.4f}",
        "100년 학습": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 승자 표시
# =========================================================

if mae_50 < mae_100:
    mae_winner = "50년 학습 모델"
elif mae_100 < mae_50:
    mae_winner = "100년 학습 모델"
else:
    mae_winner = "동일"


if mse_50 < mse_100:
    mse_winner = "50년 학습 모델"
elif mse_100 < mse_50:
    mse_winner = "100년 학습 모델"
else:
    mse_winner = "동일"


if r2_50 > r2_100:
    r2_winner = "50년 학습 모델"
elif r2_100 > r2_50:
    r2_winner = "100년 학습 모델"
else:
    r2_winner = "동일"


st.subheader("🏆 평가지표별 비교")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE가 더 낮은 모델",
        mae_winner
    )

with col2:
    st.metric(
        "MSE가 더 낮은 모델",
        mse_winner
    )

with col3:
    st.metric(
        "R²가 더 높은 모델",
        r2_winner
    )


# =========================================================
# 데이터 전체 보기
# =========================================================

with st.expander("📂 연도별 연평균 기온 데이터 확인"):

    st.dataframe(
        annual[
            [
                "연도",
                "연평균기온",
                "관측일수"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )
