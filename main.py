import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error


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
    "서울의 연평균 기온 데이터를 이용하여 "
    "1차, 3차, 9차 다항회귀 모델을 학습하고 "
    "학습에 사용하지 않은 테스트 데이터로 예측 성능을 비교합니다."
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

# 2025년까지 사용
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
# 연도를 작은 숫자로 변환
# =========================================================
# 2005년을 0으로 설정
# 예:
# 1905 → -100
# 1955 → -50
# 2005 → 0
# 2025 → 20
#
# 고차 다항식에서 큰 연도 숫자를 직접 사용하는 것을 피함
# =========================================================

annual["변환연도"] = annual["연도"] - 2005


# =========================================================
# 학습 / 테스트 데이터 분리
# =========================================================
# 중요:
# 2005년 이전 데이터만 학습에 사용
# 2005년부터는 테스트에만 사용
# =========================================================

train = annual[
    annual["연도"] < 2005
].copy()

test = annual[
    annual["연도"] >= 2005
].copy()


# =========================================================
# 데이터 개수 표시
# =========================================================

st.header("1. 훈련 데이터와 테스트 데이터")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "훈련용 연도 수",
        f"{len(train)}개"
    )
    st.caption(
        f"{train['연도'].min()}년 ~ {train['연도'].max()}년"
    )

with col2:
    st.metric(
        "테스트용 연도 수",
        f"{len(test)}개"
    )
    st.caption(
        f"{test['연도'].min()}년 ~ {test['연도'].max()}년"
    )


st.info(
    "테스트 데이터는 회귀모델을 만드는 과정에 사용하지 않고 "
    "완성된 모델의 예측 성능을 평가하는 데만 사용합니다."
)


# =========================================================
# 다항회귀 모델 생성 함수
# =========================================================

def create_polynomial_model(degree):

    model = Pipeline([
        (
            "polynomial",
            PolynomialFeatures(
                degree=degree,
                include_bias=False
            )
        ),
        (
            "linear",
            LinearRegression()
        )
    ])

    return model


# =========================================================
# 1차 / 3차 / 9차 모델 학습
# =========================================================

X_train = train[["변환연도"]]
y_train = train["연평균기온"]

X_test = test[["변환연도"]]
y_test = test["연평균기온"]


model_1 = create_polynomial_model(1)
model_3 = create_polynomial_model(3)
model_9 = create_polynomial_model(9)


# 학습은 train 데이터만 사용
model_1.fit(X_train, y_train)
model_3.fit(X_train, y_train)
model_9.fit(X_train, y_train)


# =========================================================
# 테스트 데이터 예측
# =========================================================

test["1차_예측"] = model_1.predict(X_test)
test["3차_예측"] = model_3.predict(X_test)
test["9차_예측"] = model_9.predict(X_test)


# =========================================================
# 테스트 데이터 MAE
# =========================================================
# 반드시 학습에 사용하지 않은 테스트 데이터로 계산
# =========================================================

mae_1 = mean_absolute_error(
    y_test,
    test["1차_예측"]
)

mae_3 = mean_absolute_error(
    y_test,
    test["3차_예측"]
)

mae_9 = mean_absolute_error(
    y_test,
    test["9차_예측"]
)


# =========================================================
# 2050년 예측
# =========================================================

year_2050 = np.array([[2050 - 2005]])


prediction_2050_1 = model_1.predict(
    year_2050
)[0]

prediction_2050_3 = model_3.predict(
    year_2050
)[0]

prediction_2050_9 = model_9.predict(
    year_2050
)[0]


# =========================================================
# 결과 표
# =========================================================

st.header("2. 1차·3차·9차 곡선 비교")

result = pd.DataFrame({
    "모델": [
        "1차 (직선)",
        "3차 곡선",
        "9차 곡선"
    ],
    "테스트 평균 오차 MAE (℃)": [
        mae_1,
        mae_3,
        mae_9
    ],
    "2050년 예측 기온 (℃)": [
        prediction_2050_1,
        prediction_2050_3,
        prediction_2050_9
    ]
})


st.dataframe(
    result.style.format({
        "테스트 평균 오차 MAE (℃)": "{:.3f}",
        "2050년 예측 기온 (℃)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 가장 성능이 좋은 모델
# =========================================================

mae_values = {
    "1차 (직선)": mae_1,
    "3차 곡선": mae_3,
    "9차 곡선": mae_9
}

best_model = min(
    mae_values,
    key=mae_values.get
)


st.success(
    f"테스트 데이터에서 평균 오차(MAE)가 가장 작은 모델은 "
    f"**{best_model}**입니다. "
    f"(MAE = {mae_values[best_model]:.3f}℃)"
)


# =========================================================
# 실제 데이터 + 세 가지 회귀선
# =========================================================

st.header("3. 실제 연평균 기온과 회귀곡선")

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


# 회귀곡선을 그릴 연도
curve_years = np.arange(
    annual["연도"].min(),
    2051
)

curve_x = (
    curve_years - 2005
).reshape(-1, 1)


# 1차
curve_1 = model_1.predict(curve_x)

fig.add_trace(
    go.Scatter(
        x=curve_years,
        y=curve_1,
        mode="lines",
        name="1차 회귀선"
    )
)


# 3차
curve_3 = model_3.predict(curve_x)

fig.add_trace(
    go.Scatter(
        x=curve_years,
        y=curve_3,
        mode="lines",
        name="3차 회귀곡선"
    )
)


# 9차
curve_9 = model_9.predict(curve_x)

fig.add_trace(
    go.Scatter(
        x=curve_years,
        y=curve_9,
        mode="lines",
        name="9차 회귀곡선"
    )
)


# 학습 / 테스트 구분
fig.add_vline(
    x=2005,
    line_dash="dash",
    annotation_text="2005년: 테스트 시작"
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
# 테스트 데이터에서 실제값과 예측값 비교
# =========================================================

st.header("4. 테스트 데이터 실제 기온 vs 예측 기온")

comparison = test[
    [
        "연도",
        "연평균기온",
        "1차_예측",
        "3차_예측",
        "9차_예측"
    ]
].copy()


comparison.columns = [
    "연도",
    "실제 기온 (℃)",
    "1차 예측 (℃)",
    "3차 예측 (℃)",
    "9차 예측 (℃)"
]


st.dataframe(
    comparison.style.format({
        "실제 기온 (℃)": "{:.2f}",
        "1차 예측 (℃)": "{:.2f}",
        "3차 예측 (℃)": "{:.2f}",
        "9차 예측 (℃)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 테스트 데이터 예측 그래프
# =========================================================

st.subheader("📈 테스트 기간의 실제 기온과 모델 예측")

test_fig = go.Figure()


test_fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="lines+markers",
        name="실제 기온"
    )
)


test_fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["1차_예측"],
        mode="lines+markers",
        name="1차 예측"
    )
)


test_fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["3차_예측"],
        mode="lines+markers",
        name="3차 예측"
    )
)


test_fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["9차_예측"],
        mode="lines+markers",
        name="9차 예측"
    )
)


test_fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)


st.plotly_chart(
    test_fig,
    use_container_width=True
)


# =========================================================
# 2050년 예측값 크게 표시
# =========================================================

st.header("5. 2050년 예상 기온")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "1차 모델",
        f"{prediction_2050_1:.2f} ℃"
    )

with col2:
    st.metric(
        "3차 모델",
        f"{prediction_2050_3:.2f} ℃"
    )

with col3:
    st.metric(
        "9차 모델",
        f"{prediction_2050_9:.2f} ℃"
    )


# =========================================================
# 2050년 예측값 비교 그래프
# =========================================================

st.subheader("2050년 모델별 예측값")

prediction_2050_table = pd.DataFrame({
    "모델": [
        "1차",
        "3차",
        "9차"
    ],
    "2050년 예상 기온": [
        prediction_2050_1,
        prediction_2050_3,
        prediction_2050_9
    ]
})


fig_2050 = go.Figure()

fig_2050.add_trace(
    go.Bar(
        x=prediction_2050_table["모델"],
        y=prediction_2050_table["2050년 예상 기온"],
        text=[
            f"{x:.2f}℃"
            for x in prediction_2050_table["2050년 예상 기온"]
        ],
        textposition="auto",
        name="2050년 예상 기온"
    )
)


fig_2050.update_layout(
    xaxis_title="모델",
    yaxis_title="2050년 예상 기온 (℃)"
)


st.plotly_chart(
    fig_2050,
    use_container_width=True
)


# =========================================================
# 모델 설명
# =========================================================

st.header("6. 분석 방법")

st.write(
    """
- **훈련 데이터:** 2005년 이전의 연평균 기온
- **테스트 데이터:** 2005년부터 2025년까지의 연평균 기온
- **1차:** 직선 형태의 회귀
- **3차:** 3차 다항식 형태의 회귀
- **9차:** 9차 다항식 형태의 회귀
- **평가 방법:** 학습에 사용하지 않은 테스트 데이터의 MAE
- **2050년:** 각 모델에 2050년을 입력하여 예상 기온 계산
- **연도 변환:** 계산에서는 `연도 - 2005`를 사용하여 연도 숫자의 크기를 줄임
"""
)


# =========================================================
# 연도별 데이터 확인
# =========================================================

with st.expander("📂 사용된 연평균 기온 데이터 확인"):

    st.dataframe(
        annual[
            [
                "연도",
                "연평균기온",
                "관측일수",
                "변환연도"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )
