import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── TRAIN MODEL ──────────────────────────────────────────
@st.cache_resource
def train_model():
    stats = pd.read_csv('stats.csv')
    stats = stats.drop(columns=['backward_pass', 'big_chance_missed'])
    stats = stats.fillna(stats.median(numeric_only=True))

    median_wins = stats['wins'].median()
    stats['is_top_performer'] = (stats['wins'] >= median_wins).astype(int)

    features = [
        'goals', 'total_pass', 'clean_sheet', 'goals_conceded',
        'total_tackle', 'total_scoring_att', 'ontarget_scoring_att',
        'total_cross', 'corner_taken', 'saves'
    ]

    X = stats[features]
    y = stats['is_top_performer']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    return model, features, acc, stats

# ── LOAD ─────────────────────────────────────────────────
model, features, acc, stats = train_model()

# ── UI ───────────────────────────────────────────────────
st.set_page_config(page_title="EPL Team Lookup", page_icon="⚽")
st.title("EPL Team Performance Lookup")
st.markdown("Pilih tim dan musim untuk memprediksi apakah mereka **Top Performer**.")
st.success(f"✅ Model siap! Random forrest")
st.divider()

# ── PILIH TIM & MUSIM ────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    team_list = sorted(stats['team'].unique())
    selected_team = st.selectbox("🏟️ Pilih Tim", team_list)

with col2:
    season_list = sorted(stats['season'].unique())
    selected_season = st.selectbox("📅 Pilih Musim", season_list)

# ── AMBIL DATA TIM ───────────────────────────────────────
team_data = stats[
    (stats['team'] == selected_team) &
    (stats['season'] == selected_season)
]

if team_data.empty:
    st.warning(f"⚠️ Data untuk {selected_team} di musim {selected_season} tidak tersedia.")
else:
    st.divider()
    st.subheader(f"📊 Statistik {selected_team} — {selected_season}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Goals", int(team_data['goals'].values[0]))
    col2.metric("Wins", int(team_data['wins'].values[0]))
    col3.metric("Clean Sheet", int(team_data['clean_sheet'].values[0]))
    col4.metric("Goals Conceded", int(team_data['goals_conceded'].values[0]))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Total Pass", int(team_data['total_pass'].values[0]))
    col6.metric("Total Tackle", int(team_data['total_tackle'].values[0]))
    col7.metric("Scoring Att", int(team_data['total_scoring_att'].values[0]))
    col8.metric("On Target", int(team_data['ontarget_scoring_att'].values[0]))

    st.divider()

    if st.button("🔍 Prediksi Sekarang", use_container_width=True):
        input_data = team_data[features].values

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success(f"✅ {selected_team} adalah TOP PERFORMER di musim {selected_season}!")
        else:
            st.error(f"❌ {selected_team} bukan Top Performer di musim {selected_season}.")

        st.divider()
        st.subheader("📈 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))

st.divider()
st.caption("Data: Premier League 2006–2018 | Model: Random Forest Classifier")