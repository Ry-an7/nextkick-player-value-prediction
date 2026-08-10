import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Football Player Market Value Analysis", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("market_value_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("players_model_data.csv")

model = load_model()
df = load_data()

# Sidebar

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview & Insights",
        "Potentially Undervalued Players",
        "Player Value Estimator",
    ],
)

st.sidebar.divider()

st.sidebar.markdown("### Disclaimer")

st.sidebar.caption("Model predictions are estimates based on the provided dataset and selected player characteristics.")

# Overview and Insights
if page == "Overview & Insights":
    st.title("Football Player Market Value Insights")
    st.write(
        "This application uses a machine learning model to estimate football player market values and identify players whose current market values may differ from their predicted values."
    )

    col1, col2 = st.columns(2)
    col1.metric("Players Analyzed", f"{len(df):,}")
    col2.metric("Model Used", "Gradient Boosting")

    col3, col4 = st.columns(2)
    col3.metric("Test Performance (R²)", "~63%")
    col4.metric("Top Predictive Feature", "Offensive Contribution")

    st.write("")

    st.header("Key Findings")
    st.markdown(
        """
        - **Model Performance:** Explains approximately 63% of the variation in player market value based purely on performance statistics.
        - **Main Contributors:** Offensive contribution, creativity, and successful passes are valued the highest by the model.
        - **Potential Opportunities:**  Comparing the model's predicted market values against each player's current market value identifies players who may be undervalued.
        """
    )

    st.header("What Matters Most to the Model")
    st.write(
    "This chart shows which player statistics have the greatest influence on the model's market value estimates. " \
    "Higher-ranked statistics have a stronger influence on the model's valuation."
    )
    importance_df = (
        pd.DataFrame({
                "feature": model.feature_names_in_,
                "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .head(10)
    )

    importance_df["feature"] = (importance_df["feature"].str.replace("_", " ").str.title())
    st.bar_chart(importance_df.set_index("feature"))

# Undervalued Players
elif page == "Potentially Undervalued Players":
    st.title("Potentially Undervalued Players")
    st.write("Explore players whose model-predicted market values are higher than their current market valuations. " \
    "   The value gap shows the difference between the model's estimated value and the player's current market value, with larger gaps highlighting potential scouting opportunities.")

    features = [
        "age",
        "position",
        "minutes_played",
        "goals",
        "assists",
        "offensive_contribution",
        "creativity_score",
        "successful_passes",
        "defensive_actions",
        "save_percentage",
        "player_rating",
    ]

    X_players = pd.get_dummies(df[features].copy(), columns=["position"], drop_first=True)
    for col in model.feature_names_in_:
        if col not in X_players.columns:
            X_players[col] = 0
    X_players = X_players[model.feature_names_in_]

    predictions = model.predict(X_players)
    results = df.copy()
    results["predicted_market_value"] = np.exp(predictions)
    results["value_difference"] = (
        results["predicted_market_value"] - results["market_value_eur"]
    )

    undervalued_players = results[results["value_difference"] > 0].sort_values("value_difference", ascending=False)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_position = st.selectbox(
            "Filter by Position",
            ["All"] + sorted(df["position"].unique()),
            key="pos_filter",
        )
    with col_f2:
        top_n = st.slider(
            "Number of players to display",
            min_value=5,
            max_value=100,
            value=20,
        )

    filtered_players = undervalued_players.copy()
    if selected_position != "All":
        filtered_players = filtered_players[
            filtered_players["position"] == selected_position
        ]

    display_df = (
        filtered_players[
            [
                "player_name",
                "position",
                "market_value_eur",
                "predicted_market_value",
                "value_difference",
            ]
        ]
        .head(top_n)
        .copy()
    )

    for col in [
        "market_value_eur",
        "predicted_market_value",
        "value_difference",
    ]:
        display_df[col] = (display_df[col] / 1_000_000).round(2)

    display_df = display_df.rename(
        columns={
            "market_value_eur": "Current Value (€M)",
            "predicted_market_value": "Predicted Value (€M)",
            "value_difference": "Value Gap (€M)",
        }
    )

    table_tab, chart_tab = st.tabs(
        ["Player Details", "Value Gap Chart"]
    )

    with table_tab:
        st.dataframe(
            display_df,
            use_container_width=True
        )

    with chart_tab:
        chart_data = (
            display_df
            .set_index("player_name")["Value Gap (€M)"]
            .sort_values()
        )

        st.bar_chart(
            chart_data,
            horizontal=True
        )

        st.caption(
            "Players are ranked by estimated value gap. "
            "A larger bar indicates a greater difference between estimated "
            "and current market value."
        )


# Player Value Esitmator
elif page == "Player Value Estimator":
    st.title("Player Value Estimator")
    st.write("Adjust player statistics to generate a model-based market valuation. " \
    "Change key performance characteristics to see how different player profiles affect the estimated value.")

    col1, col2, col3 = st.columns(3)

    with col1:
        position = st.selectbox("Position", sorted(df["position"].unique()))
        age = st.slider(
            "Age",
            int(df["age"].min()),
            int(df["age"].max()),
            int(df["age"].median()),
        )
        player_rating = st.slider(
            "Player Rating",
            float(df["player_rating"].min()),
            float(df["player_rating"].max()),
            float(df["player_rating"].median()),
        )

    with col2:
        offensive_contribution = st.slider(
            "Offensive Contribution",
            float(df["offensive_contribution"].min()),
            float(df["offensive_contribution"].max()),
            float(df["offensive_contribution"].median()),
        )
        creativity_score = st.slider(
            "Creativity Score",
            float(df["creativity_score"].min()),
            float(df["creativity_score"].max()),
            float(df["creativity_score"].median()),
        )

    with col3:
        successful_passes = st.slider(
            "Successful Passes",
            int(df["successful_passes"].min()),
            int(df["successful_passes"].max()),
            int(df["successful_passes"].median()),
        )

        if position == "Goalkeeper":
            save_percentage = st.slider(
                "Save Percentage",
                float(df["save_percentage"].min()),
                float(df["save_percentage"].max()),
                float(df["save_percentage"].median()),
            )
        else:
            save_percentage = float(df["save_percentage"].median())

    with st.expander("Advanced Statistics"):
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            minutes_played = st.slider(
                "Minutes Played",
                int(df["minutes_played"].min()),
                int(df["minutes_played"].max()),
                int(df["minutes_played"].median()),
            )
            goals = st.slider(
                "Goals",
                int(df["goals"].min()),
                int(df["goals"].max()),
                int(df["goals"].median()),
            )
        with col_a2:
            assists = st.slider(
                "Assists",
                int(df["assists"].min()),
                int(df["assists"].max()),
                int(df["assists"].median()),
            )
            defensive_actions = st.slider(
                "Defensive Actions",
                int(df["defensive_actions"].min()),
                int(df["defensive_actions"].max()),
                int(df["defensive_actions"].median()),
            )

    st.write("")
    if st.button("Estimate Market Value", type="primary"):
        input_data = {
            "age": age,
            "minutes_played": minutes_played,
            "goals": goals,
            "assists": assists,
            "offensive_contribution": offensive_contribution,
            "creativity_score": creativity_score,
            "successful_passes": successful_passes,
            "defensive_actions": defensive_actions,
            "save_percentage": save_percentage,
            "player_rating": player_rating,
        }

        input_df = pd.DataFrame([input_data])

        for col in model.feature_names_in_:
            if col.startswith("position_"):
                pos_name = col.replace("position_", "")
                input_df[col] = 1 if position == pos_name else 0
            elif col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[model.feature_names_in_]

        prediction = model.predict(input_df)[0]
        predicted_value = np.exp(prediction)

        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Estimated Market Value", f"€{predicted_value:,.0f}")
        res_col2.metric("Value in Millions (€M)", f"€{predicted_value / 1_000_000:.2f}M")