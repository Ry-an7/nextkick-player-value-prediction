import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Football Player Market Value Analysis", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("market_value_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("players_model_data.csv")

model = load_model()
df = load_data()

# Overview
st.title("Football Player Market Value Insights")
st.write(
    """
    This application uses a machine learning model to estimate football player market values and identify players whose current market values may differ from their predicted values.
    """
)
st.write("")
st.write("")

st.header("Project Objective")
st.write(
    """
    The goal of this project was to develop a model capable of predicting football player market values based on player performance characteristics.
    These predictions were then compared with current market values from the dataset to identify players whose estimated values may differ from their current valuations.
    """
)
st.write("")
st.write("")

st.header("Project Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Players Analyzed", "1,248")
col2.metric("Model Used", "Gradient Boosting")
col3.metric("Test Performance (R²)", "0.63")
col4.metric("Top Predictive Feature", "Offensive Contribution")
st.write("")
st.write("")

st.header("Data Overview")
st.write(
    """
    The analysis used a player-level dataset containing football performance statistics and market values from the 2026 FIFA World Cup tournament. 
    Player statistics were aggregated to create a single profile for each player, allowing the model to learn the relationships between player performance and market value.
    """
)
st.write("")
st.write("")

st.subheader("What Matters Most to the Model")
st.write(
    "These are the statistics that influence the model's value estimate the most."
)

importance_df = pd.DataFrame({
    "feature": model.feature_names_in_,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False).head(10)

importance_df["feature"] = importance_df["feature"].str.replace("_", " ").str.title()

st.bar_chart(importance_df.set_index("feature"))
st.write("")
st.write("")

st.header("Key Findings")
st.markdown(
    """
    ### Model Performance

    The model explains approximately 63% of the variation in player market value based on performance statistics alone.
    This means that player performance is a strong factor in market value, but not the only one.
    Factors outside of this dataset, such as club reputation, contract situation, and transfer market conditions,   also influence market value but are not captured by this model.

    ### Main Factors Influencing Value

    Offensive contribution was the strongest predictor of market value,followed by creativity score and successful passes.
    Players who create and finish attacking chances tend to be valued highest by the model, which is consistent with how transfer markets generally value attacking talent.
    
    ### Identifying Potential Opportunities

    Comparing the model's predicted market values against each player's current market value identifies players who may be undervalued.
    These are cases where their perforance suggests a higher value than the market currently reflects, making them worth a closer scouting look.
    """
)
st.write("")
st.write("")

st.header("How to Use This Tool")
st.markdown(
    """
    This tool is divided into two tabs, each designed for a different purpose.

    ### Tab 1: Identify Potentially Undervalued Players

    Explore players whose model-predicted market values are higher than their current market values. 
    Players are ranked by the estimated value gap, with larger positive differences indicating greater potential undervaluation.

    ### Tab 2: Estimate Player Value

    Explore how different player characteristics influence estimated market value. 
    Adjust key performance statistics to generated a model-based valuation for different player profiles.

    Additional statistics are available under **Advanced Player Statistics** for a mroe detailed estimate.
    """
)
st.write("")
st.write("")

st.header("Explore the Model")
tab1, tab2 = st.tabs(
    [
        "Potentially Undervalued Players",
        "Player Value Estimator"
    ]
)

# Potentially Undervalued Players
with tab1:
    st.subheader("Potentially Undervalued Players")
    st.write(
        """
        These players have a higher predicted market value than their current market value.
        Use the filters below to explore potential opportunities by position and number of players.
        """
    )

    # Prepare player data for predictions
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
        "player_rating"
    ]

    X_players = df[features].copy()

    X_players = pd.get_dummies(
        X_players,
        columns=["position"],
        drop_first=True
    )

    for col in model.feature_names_in_:
        if col not in X_players.columns:
            X_players[col] = 0

    X_players = X_players[
        model.feature_names_in_
    ]

    # Generate predictions
    predictions = model.predict(X_players)

    results = df.copy()

    results["predicted_market_value"] = np.exp(predictions)

    results["value_difference"] = (
        results["predicted_market_value"]
        - results["market_value_eur"]
    )

    # Keep only potentially undervalued players
    undervalued_players = results[
        results["value_difference"] > 0
    ].sort_values(
        "value_difference",
        ascending=False
    )

    # Filters
    selected_position = st.selectbox(
        "Filter by Position",
        ["All"] + sorted(df["position"].unique()),
        key="position_filter"
    )

    top_n = st.slider(
        "Number of players to display",
        min_value=5,
        max_value=100,
        value=20
    )

    # Apply position filter
    filtered_players = undervalued_players.copy()

    if selected_position != "All":
        filtered_players = filtered_players[
            filtered_players["position"] == selected_position
        ]

    # Select columns and number of players
    display_df = filtered_players[
        [
            "player_name",
            "position",
            "market_value_eur",
            "predicted_market_value",
            "value_difference"
        ]
    ].head(top_n).copy()

    # Convert values to millions
    for col in [
        "market_value_eur",
        "predicted_market_value",
        "value_difference"
    ]:
        display_df[col] = (
            display_df[col] / 1_000_000
        ).round(2)

    # Rename columns
    display_df = display_df.rename(
        columns={
            "market_value_eur": "Current Value (€M)",
            "predicted_market_value": "Predicted Value (€M)",
            "value_difference": "Value Gap (€M)"
        }
    )

    # Chart and table tabs
    chart_tab, table_tab = st.tabs(
        ["Value Gap Chart", "Player Details"]
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

    with table_tab:
        st.dataframe(
            display_df,
            use_container_width=True
        )
# with tab1:
#     st.subheader("Potentially Undervalued Players")
#     st.write(
#     """
#         These players have a higher predicted market value than their current market value.
#         Use the filters below to explore potential opportunities by position and number of players.
#     """
#     )

#     features = [
#         "age",
#         "position",
#         "minutes_played",
#         "goals",
#         "assists",
#         "offensive_contribution",
#         "creativity_score",
#         "successful_passes",
#         "defensive_actions",
#         "save_percentage",
#         "player_rating"
#     ]

#     X_players = df[features].copy()

#     X_players = pd.get_dummies(
#         X_players,
#         columns=["position"],
#         drop_first=True
#     )

#     for col in model.feature_names_in_:
#         if col not in X_players.columns:
#             X_players[col] = 0

#     X_players = X_players[
#         model.feature_names_in_
#     ]

#     predictions = model.predict(X_players)
#     results = df.copy()

#     results["predicted_market_value"] = (
#         np.exp(predictions)
#     )

#     results["value_difference"] = (
#         results["predicted_market_value"]
#         -
#         results["market_value_eur"]
#     )

#     undervalued_players = results[
#         results["value_difference"] > 0
#     ]

#     undervalued_players = undervalued_players.sort_values(
#         "value_difference",
#         ascending=False
#     )

#     selected_position = st.selectbox(
#         "Filter by Position",
#         ["All"] + sorted(df["position"].unique()),
#         key="position_filter"
#     )

#     if selected_position != "All":
#         undervalued_players = undervalued_players[
#             undervalued_players["position"] == selected_position
#         ]

#     top_n = st.slider(
#     "Number of players to display",
#     min_value=5,
#     max_value=100,
#     value=20
#     )

#     display_df = undervalued_players[
#     [
#         "player_name",
#         "position",
#         "market_value_eur",
#         "predicted_market_value",
#         "value_difference"
#     ]
#     ].head(top_n).copy()


#     for col in [
#         "market_value_eur",
#         "predicted_market_value",
#         "value_difference"
#     ]:
#         display_df[col] = (
#             display_df[col] / 1_000_000
#         ).round(2)

#     display_df = display_df.rename(
#         columns={
#             "market_value_eur": "Current Value (€M)",
#             "predicted_market_value": "Predicted Value (€M)",
#             "value_difference": "Value Gap (€M)"
#         }
#     )

#     st.write("")
#     chart_data = display_df.set_index("player_name")["Value Gap (€M)"].sort_values()
#     st.bar_chart(chart_data, horizontal=True)
#     st.caption("Players are ranked by estimated value gap. A larger bar indicates a greater difference between estimated and current market value.")

#     st.dataframe(display_df)


# Player Value Estimator
with tab2:
    st.subheader("Estimate Player Market Value")
    st.write(
        """
        Adjust key player characteristics to estimate market value.
        Additional statistics can be adjusted in the advanced options.
        """
    )

    # Main Statistics
    position = st.selectbox(
        "Position",
        sorted(df["position"].unique())
    )
    age = st.slider(
        "Age",
        int(df["age"].min()),
        int(df["age"].max()),
        int(df["age"].median())
    )
    offensive_contribution = st.slider(
        "Offensive Contribution",
        float(df["offensive_contribution"].min()),
        float(df["offensive_contribution"].max()),
        float(df["offensive_contribution"].median())
    )
    creativity_score = st.slider(
        "Creativity Score",
        float(df["creativity_score"].min()),
        float(df["creativity_score"].max()),
        float(df["creativity_score"].median())
    )
    successful_passes = st.slider(
        "Successful Passes",
        int(df["successful_passes"].min()),
        int(df["successful_passes"].max()),
        int(df["successful_passes"].median())
    )
    player_rating = st.slider(
        "Player Rating",
        float(df["player_rating"].min()),
        float(df["player_rating"].max()),
        float(df["player_rating"].median())
    )

    # Advanced Statistics
    default_save_percentage = float(df["save_percentage"].median())

    with st.expander("Advanced Player Statistics"):
        minutes_played = st.slider(
            "Minutes Played",
            int(df["minutes_played"].min()),
            int(df["minutes_played"].max()),
            int(df["minutes_played"].median())
        )
        goals = st.slider(
            "Goals",
            int(df["goals"].min()),
            int(df["goals"].max()),
            int(df["goals"].median())
        )
        assists = st.slider(
            "Assists",
            int(df["assists"].min()),
            int(df["assists"].max()),
            int(df["assists"].median())
        )
        defensive_actions = st.slider(
            "Defensive Actions",
            int(df["defensive_actions"].min()),
            int(df["defensive_actions"].max()),
            int(df["defensive_actions"].median())
        )
        save_percentage = st.slider(
            "Save Percentage",
            float(df["save_percentage"].min()),
            float(df["save_percentage"].max()),
            default_save_percentage
        )
        if position != "Goalkeeper" and save_percentage != default_save_percentage:
            st.warning(
                "Save Percentage only affects goalkeepers. "
                "This value will not affect any other predictions."
            )

    # Prediction
    if st.button("Estimate Market Value"):

        input_data = {
            "age": age,
            "position": position,
            "minutes_played": minutes_played,
            "goals": goals,
            "assists": assists,
            "offensive_contribution": offensive_contribution,
            "creativity_score": creativity_score,
            "successful_passes": successful_passes,
            "defensive_actions": defensive_actions,
            "save_percentage": (save_percentage if position == "Goalkeeper" else default_save_percentage),
            "player_rating": player_rating
        }

        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(
            input_df,
            columns=["position"],
            drop_first=True
        )

        for col in model.feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[
            model.feature_names_in_
        ]

        prediction = model.predict(input_df)[0]
        predicted_value = np.exp(prediction)
        st.success(f"Estimated Market Value: €{predicted_value:,.0f}")

st.divider()
st.caption(
    """
    Disclaimer: Model predictions are estimates based on the provided dataset and selected player characteristics. The tool is intended to support player evaluation and should not replace expert judgement or current market analysis.
    """
)