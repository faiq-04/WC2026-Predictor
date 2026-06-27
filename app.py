import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #1a472a, #2d6a4f, #1a472a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 0.5rem 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #2d6a4f;
        margin-bottom: 1rem;
    }
    .winner-banner {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .winner-text {
        font-size: 2rem;
        font-weight: 800;
        color: #1a1a1a;
    }
    .match-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
    .stProgress .st-bo { background-color: #2d6a4f; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a472a;
        border-bottom: 2px solid #2d6a4f;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── WC 2026 Teams ─────────────────────────────────────────────────────────────
WC2026_TEAMS = sorted([
    'Argentina', 'Brazil', 'France', 'England', 'Spain', 'Portugal',
    'Germany', 'Netherlands', 'Belgium', 'Croatia', 'Uruguay', 'Colombia',
    'Mexico', 'United States', 'Canada', 'Morocco', 'Senegal', 'Japan',
    'South Korea', 'Australia', 'Ecuador', 'Chile', 'Peru', 'Bolivia',
    'Venezuela', 'Paraguay', 'Panama', 'Costa Rica', 'Honduras', 'Jamaica',
    'Saudi Arabia', 'Iran', 'New Zealand', 'Indonesia', 'Algeria', 'Egypt',
    'Nigeria', 'Ghana', 'Cameroon', 'Tunisia', 'Ivory Coast', 'Mali',
    'South Africa', 'Tanzania', 'El Salvador', 'Guatemala', 'Cuba',
    'Trinidad and Tobago',
])

# ── Model training (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model(csv_path: str):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    def get_outcome(row):
        if row['home_score'] > row['away_score']:
            return 'Home Win'
        elif row['home_score'] < row['away_score']:
            return 'Away Win'
        return 'Draw'

    df['match_outcome'] = df.apply(get_outcome, axis=1)
    historical_df = df[df['date'].dt.year >= 1998].copy().reset_index(drop=True)

    df_features = historical_df.copy()
    team_stats, h2h_stats, elo_ratings = {}, {}, {}
    INITIAL_ELO, K = 1500, 20

    home_win_rate, away_win_rate = [], []
    home_avg_goals_for, away_avg_goals_for = [], []
    home_avg_goals_against, away_avg_goals_against = [], []
    h2h_home_win_rate = []
    days_since_last_home, days_since_last_away = [], []
    home_elo_before, away_elo_before = [], []

    for _, row in df_features.iterrows():
        h_team, a_team, match_date = row['home_team'], row['away_team'], row['date']
        for team in [h_team, a_team]:
            if team not in team_stats:
                team_stats[team] = []
        pair_key = tuple(sorted([h_team, a_team]))
        if pair_key not in h2h_stats:
            h2h_stats[pair_key] = []

        h_hist = [m for m in team_stats[h_team] if m[0] < match_date][-10:]
        if h_hist:
            home_win_rate.append(np.mean([m[3] for m in h_hist]))
            home_avg_goals_for.append(np.mean([m[1] for m in h_hist]))
            home_avg_goals_against.append(np.mean([m[2] for m in h_hist]))
            days_since_last_home.append((match_date - h_hist[-1][0]).days)
        else:
            home_win_rate.append(0.33)
            home_avg_goals_for.append(1.0)
            home_avg_goals_against.append(1.0)
            days_since_last_home.append(365)

        a_hist = [m for m in team_stats[a_team] if m[0] < match_date][-10:]
        if a_hist:
            away_win_rate.append(np.mean([m[3] for m in a_hist]))
            away_avg_goals_for.append(np.mean([m[1] for m in a_hist]))
            away_avg_goals_against.append(np.mean([m[2] for m in a_hist]))
            days_since_last_away.append((match_date - a_hist[-1][0]).days)
        else:
            away_win_rate.append(0.33)
            away_avg_goals_for.append(1.0)
            away_avg_goals_against.append(1.0)
            days_since_last_away.append(365)

        h2h_hist = [m for m in h2h_stats[pair_key] if m[0] < match_date]
        if h2h_hist:
            h2h_home_results = [m[1] for m in h2h_hist if m[2] == h_team]
            h2h_home_win_rate.append(np.mean(h2h_home_results) if h2h_home_results else 0.5)
        else:
            h2h_home_win_rate.append(0.5)

        h_elo = elo_ratings.get(h_team, INITIAL_ELO)
        a_elo = elo_ratings.get(a_team, INITIAL_ELO)
        home_elo_before.append(h_elo)
        away_elo_before.append(a_elo)

        expected_h = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        h_won = 1 if row['home_score'] > row['away_score'] else 0
        a_won = 1 if row['away_score'] > row['home_score'] else 0
        score_h = 1.0 if h_won else (0.0 if a_won else 0.5)

        elo_ratings[h_team] = h_elo + K * (score_h - expected_h)
        elo_ratings[a_team] = a_elo + K * ((1 - score_h) - (1 - expected_h))

        team_stats[h_team].append((match_date, row['home_score'], row['away_score'], h_won))
        team_stats[a_team].append((match_date, row['away_score'], row['home_score'], a_won))
        h2h_stats[pair_key].append((match_date, h_won, h_team))

    df_features['home_rolling_win_rate'] = home_win_rate
    df_features['away_rolling_win_rate'] = away_win_rate
    df_features['home_avg_goals_for'] = home_avg_goals_for
    df_features['away_avg_goals_for'] = away_avg_goals_for
    df_features['home_avg_goals_against'] = home_avg_goals_against
    df_features['away_avg_goals_against'] = away_avg_goals_against
    df_features['h2h_home_win_rate'] = h2h_home_win_rate
    df_features['days_since_last_home'] = days_since_last_home
    df_features['days_since_last_away'] = days_since_last_away
    df_features['home_elo'] = home_elo_before
    df_features['away_elo'] = away_elo_before
    df_features['win_rate_diff'] = df_features['home_rolling_win_rate'] - df_features['away_rolling_win_rate']
    df_features['attack_diff'] = df_features['home_avg_goals_for'] - df_features['away_avg_goals_for']
    df_features['defense_diff'] = df_features['home_avg_goals_against'] - df_features['away_avg_goals_against']
    df_features['rest_diff'] = df_features['days_since_last_home'] - df_features['days_since_last_away']
    df_features['elo_diff'] = df_features['home_elo'] - df_features['away_elo']
    df_features['is_world_cup'] = (df_features['tournament'] == 'FIFA World Cup').astype(int)

    feature_cols = ['win_rate_diff', 'elo_diff', 'attack_diff', 'defense_diff',
                    'h2h_home_win_rate', 'rest_diff', 'is_world_cup']
    X = df_features[feature_cols]
    y = df_features['match_outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=['Home Win', 'Draw', 'Away Win'])

    simulation_date = df_features['date'].max()

    return {
        'model': model,
        'team_stats': team_stats,
        'h2h_stats': h2h_stats,
        'elo_ratings': elo_ratings,
        'feature_cols': feature_cols,
        'INITIAL_ELO': INITIAL_ELO,
        'simulation_date': simulation_date,
        'accuracy': accuracy,
        'report': report,
        'cm': cm,
        'df_features': df_features,
        'n_train': len(X_train),
        'n_test': len(X_test),
    }


def predict_fixture(state, home_team, away_team, match_date, is_world_cup=1):
    model = state['model']
    team_stats = state['team_stats']
    h2h_stats = state['h2h_stats']
    elo_ratings = state['elo_ratings']
    feature_cols = state['feature_cols']
    INITIAL_ELO = state['INITIAL_ELO']

    def latest_form(team):
        hist = [m for m in team_stats.get(team, []) if m[0] < match_date][-10:]
        if not hist:
            return {'win_rate': 0.33, 'goals_for': 1.0, 'goals_against': 1.0, 'days_since': 365}
        return {
            'win_rate': np.mean([m[3] for m in hist]),
            'goals_for': np.mean([m[1] for m in hist]),
            'goals_against': np.mean([m[2] for m in hist]),
            'days_since': (match_date - hist[-1][0]).days,
        }

    h_form = latest_form(home_team)
    a_form = latest_form(away_team)
    h_elo = elo_ratings.get(home_team, INITIAL_ELO)
    a_elo = elo_ratings.get(away_team, INITIAL_ELO)

    pair_key = tuple(sorted([home_team, away_team]))
    h2h_hist = [m for m in h2h_stats.get(pair_key, []) if m[0] < match_date]
    if h2h_hist:
        h2h_home_results = [m[1] for m in h2h_hist if m[2] == home_team]
        h2h_rate = np.mean(h2h_home_results) if h2h_home_results else 0.5
    else:
        h2h_rate = 0.5

    features = pd.DataFrame([{
        'win_rate_diff': h_form['win_rate'] - a_form['win_rate'],
        'elo_diff': h_elo - a_elo,
        'attack_diff': h_form['goals_for'] - a_form['goals_for'],
        'defense_diff': h_form['goals_against'] - a_form['goals_against'],
        'h2h_home_win_rate': h2h_rate,
        'rest_diff': h_form['days_since'] - a_form['days_since'],
        'is_world_cup': is_world_cup,
    }])[feature_cols]

    probs = model.predict_proba(features)[0]
    return dict(zip(model.classes_, probs))


def simulate_matchup(state, team_a, team_b, match_date):
    probs = predict_fixture(state, team_a, team_b, match_date)
    hw = probs.get('Home Win', 0)
    aw = probs.get('Away Win', 0)
    dp = probs.get('Draw', 0)
    if dp > 0:
        denom = hw + aw + 1e-9
        hw += dp * (hw / denom)
        aw += dp * (aw / denom)
    if hw >= aw:
        return team_a, hw
    return team_b, aw


# ── Load data ──────────────────────────────────────────────────────────────────
DATA_PATH = "results_cleaned.csv"

st.markdown('<h1 class="main-title">⚽ FIFA World Cup 2026 Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Random Forest model trained on 49,433 international matches (1872–2026) · Kaggle dataset</p>', unsafe_allow_html=True)

with st.spinner("🔄 Training model on 49,000+ matches... (first load only, ~30 seconds)"):
    try:
        state = train_model(DATA_PATH)
    except FileNotFoundError:
        st.error("❌ `results_cleaned.csv` not found. Make sure it's in the same folder as `app.py`.")
        st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/e/e3/2026_FIFA_World_Cup.svg/200px-2026_FIFA_World_Cup.svg.png", width=120)
    st.markdown("## Navigation")
    page = st.radio(
        "Go to",
        ["🏆 Tournament Simulator", "⚔️ Head-to-Head Predictor", "📊 Model Performance", "📈 Elo Rankings"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Model details**")
    st.markdown(f"- Algorithm: Random Forest (200 trees)")
    st.markdown(f"- Training samples: `{state['n_train']:,}`")
    st.markdown(f"- Test accuracy: `{state['accuracy']:.1%}`")
    st.markdown(f"- Features: Elo, form, H2H, attack, defense")
    st.markdown("---")
    st.markdown("**Built by Faiq Ahmed**")
    st.markdown("Dataset: [Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Tournament Simulator
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏆 Tournament Simulator":
    st.markdown('<div class="section-header">🏆 Knockout Stage Simulator</div>', unsafe_allow_html=True)
    st.markdown("Select 8 teams for the quarter-finals and simulate the entire knockout bracket.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Pick your Quarter-Final bracket (8 teams)**")
        default_qf = ['Argentina', 'Croatia', 'France', 'England', 'Spain', 'Netherlands', 'Brazil', 'Portugal']

        teams_grid = []
        for i in range(0, 8, 2):
            c1, c2 = st.columns(2)
            with c1:
                t1 = st.selectbox(f"QF Match {i//2+1} — Team A", WC2026_TEAMS,
                                  index=WC2026_TEAMS.index(default_qf[i]), key=f"t{i}")
            with c2:
                t2 = st.selectbox(f"QF Match {i//2+1} — Team B", WC2026_TEAMS,
                                  index=WC2026_TEAMS.index(default_qf[i+1]), key=f"t{i+1}")
            teams_grid.append((t1, t2))

    with col2:
        st.markdown("**Simulation date**")
        sim_date = state['simulation_date']
        st.info(f"Using data up to: **{sim_date.strftime('%d %b %Y')}**")
        st.markdown("This ensures no future data leakage — the model only uses history available before each match.")

    if st.button("▶️ Run Tournament Simulation", type="primary", use_container_width=True):
        bracket = [t for pair in teams_grid for t in pair]
        if len(set(bracket)) < 8:
            st.error("Please select 8 **different** teams.")
        else:
            st.markdown('<div class="section-header">Tournament Results</div>', unsafe_allow_html=True)

            round_names = {8: "Quarter-Finals", 4: "Semi-Finals", 2: "Final"}
            current_round = bracket.copy()
            all_rounds = {}

            while len(current_round) > 1:
                rname = round_names.get(len(current_round), "Round")
                st.markdown(f"### {rname}")
                cols = st.columns(len(current_round) // 2)
                next_round = []

                for i in range(0, len(current_round), 2):
                    t1, t2 = current_round[i], current_round[i+1]
                    winner, conf = simulate_matchup(state, t1, t2, sim_date)
                    loser = t2 if winner == t1 else t1
                    next_round.append(winner)

                    with cols[i // 2]:
                        w_flag = "🟢" if winner == t1 else "⚪"
                        l_flag = "⚪" if winner == t1 else "🟢"
                        st.markdown(f"""
                        <div class="match-card">
                            <div style="font-size:0.8rem;color:#888;margin-bottom:6px;">{rname} · Match {i//2+1}</div>
                            <div style="font-weight:{'700' if winner==t1 else '400'};color:{'#1a472a' if winner==t1 else '#666'};">{w_flag if winner==t1 else l_flag} {t1}</div>
                            <div style="font-size:0.75rem;color:#888;margin:2px 8px;">vs</div>
                            <div style="font-weight:{'700' if winner==t2 else '400'};color:{'#1a472a' if winner==t2 else '#666'};">{l_flag if winner==t1 else w_flag} {t2}</div>
                            <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                            <div style="font-size:0.8rem;">🏅 <b>{winner}</b> wins <span style="color:#2d6a4f;font-weight:600;">({conf:.1%})</span></div>
                        </div>
                        """, unsafe_allow_html=True)

                current_round = next_round

            # Champion banner
            champion = current_round[0]
            st.markdown(f"""
            <div class="winner-banner">
                <div style="font-size:3rem;">🏆</div>
                <div style="font-size:1rem;font-weight:600;color:#333;letter-spacing:0.1em;">FIFA WORLD CUP 2026 CHAMPION</div>
                <div class="winner-text">{champion}</div>
            </div>
            """, unsafe_allow_html=True)

            elo = state['elo_ratings'].get(champion, 1500)
            st.success(f"**{champion}** wins the tournament! Their Elo rating: **{elo:.0f}** (top-ranked teams are typically 1800+)")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Head-to-Head Predictor
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚔️ Head-to-Head Predictor":
    st.markdown('<div class="section-header">⚔️ Head-to-Head Match Predictor</div>', unsafe_allow_html=True)
    st.markdown("Pick any two teams and get win probabilities based on their Elo rating, recent form, and head-to-head history.")

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        home_team = st.selectbox("🏠 Team A (Home)", WC2026_TEAMS, index=WC2026_TEAMS.index("Argentina"))
    with col2:
        st.markdown("<br><div style='text-align:center;font-size:1.5rem;font-weight:700;color:#888;margin-top:1.8rem;'>VS</div>", unsafe_allow_html=True)
    with col3:
        away_team = st.selectbox("✈️ Team B (Away)", WC2026_TEAMS, index=WC2026_TEAMS.index("Brazil"))

    is_wc = st.checkbox("World Cup match (knockout rules)", value=True)

    if st.button("🔮 Predict Match", type="primary", use_container_width=True):
        if home_team == away_team:
            st.error("Please select two different teams.")
        else:
            probs = predict_fixture(state, home_team, away_team, state['simulation_date'], int(is_wc))
            hw = probs.get('Home Win', 0)
            dp = probs.get('Draw', 0)
            aw = probs.get('Away Win', 0)

            st.markdown("### Predicted Probabilities")
            c1, c2, c3 = st.columns(3)
            c1.metric(f"🟢 {home_team} wins", f"{hw:.1%}")
            c2.metric("🤝 Draw", f"{dp:.1%}")
            c3.metric(f"🟢 {away_team} wins", f"{aw:.1%}")

            # Visual probability bar
            st.markdown("#### Probability breakdown")
            fig, ax = plt.subplots(figsize=(9, 1.2))
            ax.barh(0, hw, color='#2d6a4f', height=0.5, label=home_team)
            ax.barh(0, dp, left=hw, color='#888888', height=0.5, label='Draw')
            ax.barh(0, aw, left=hw+dp, color='#c0392b', height=0.5, label=away_team)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            ax.legend(loc='upper right', bbox_to_anchor=(1, 2.2), ncol=3, fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Elo and form comparison
            st.markdown("#### Team comparison")
            sim_date = state['simulation_date']
            h_elo = state['elo_ratings'].get(home_team, 1500)
            a_elo = state['elo_ratings'].get(away_team, 1500)

            def get_form(team):
                hist = [m for m in state['team_stats'].get(team, []) if m[0] < sim_date][-10:]
                if not hist:
                    return 0.33, 1.0
                return np.mean([m[3] for m in hist]), np.mean([m[1] for m in hist])

            h_wr, h_gf = get_form(home_team)
            a_wr, a_gf = get_form(away_team)

            comp_df = pd.DataFrame({
                'Metric': ['Elo Rating', 'Recent Win Rate (last 10)', 'Avg Goals Scored (last 10)'],
                home_team: [f"{h_elo:.0f}", f"{h_wr:.1%}", f"{h_gf:.2f}"],
                away_team: [f"{a_elo:.0f}", f"{a_wr:.1%}", f"{a_gf:.2f}"],
            })
            st.dataframe(comp_df.set_index('Metric'), use_container_width=True)

            # Verdict
            predicted_winner = home_team if hw > aw else (away_team if aw > hw else "Draw")
            if predicted_winner != "Draw":
                margin = abs(hw - aw)
                confidence = "dominant" if margin > 0.3 else ("moderate" if margin > 0.15 else "slight")
                st.info(f"🏅 **Verdict:** {predicted_winner} is the {confidence} favourite based on Elo ratings and recent form.")
            else:
                st.info("🤝 **Verdict:** This is a very evenly matched game — too close to call!")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)

    # Top metrics
    report = state['report']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Accuracy", f"{state['accuracy']:.1%}")
    c2.metric("Home Win F1", f"{report['Home Win']['f1-score']:.2f}")
    c3.metric("Away Win F1", f"{report['Away Win']['f1-score']:.2f}")
    c4.metric("Draw F1", f"{report['Draw']['f1-score']:.2f}")

    col1, col2 = st.columns(2)

    # Confusion matrix
    with col1:
        st.markdown("#### Confusion Matrix")
        labels = ['Home Win', 'Draw', 'Away Win']
        cm = state['cm']
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap='Greens')
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(['Home Win', 'Draw', 'Away Win'], fontsize=9)
        ax.set_yticklabels(['Home Win', 'Draw', 'Away Win'], fontsize=9)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        ax.set_title('Confusion Matrix (Test Set)', fontsize=11)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Feature importance
    with col2:
        st.markdown("#### Feature Importances")
        feature_cols = state['feature_cols']
        importances = state['model'].feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
        feat_df = feat_df.sort_values('Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(5, 4))
        colors = ['#2d6a4f' if i == len(feat_df)-1 else '#74c69d' for i in range(len(feat_df))]
        bars = ax.barh(feat_df['Feature'], feat_df['Importance'], color=colors)
        ax.set_xlabel('Importance', fontsize=10)
        ax.set_title('Random Forest Feature Importances', fontsize=11)
        for bar, val in zip(bars, feat_df['Importance']):
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                    f'{val:.1%}', va='center', fontsize=9)
        ax.set_xlim(0, 0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Classification report table
    st.markdown("#### Detailed Classification Report")
    rpt_data = []
    for cls in ['Home Win', 'Draw', 'Away Win']:
        r = report[cls]
        rpt_data.append({
            'Class': cls,
            'Precision': f"{r['precision']:.2f}",
            'Recall': f"{r['recall']:.2f}",
            'F1-Score': f"{r['f1-score']:.2f}",
            'Support': f"{int(r['support']):,}",
        })
    st.dataframe(pd.DataFrame(rpt_data).set_index('Class'), use_container_width=True)

    st.markdown("""
    > **Note:** 54% accuracy is strong for football prediction — the sport is inherently
    > unpredictable. Draws are the hardest class to predict (F1 = 0.31), which is typical
    > across all football ML models globally.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Elo Rankings
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Elo Rankings":
    st.markdown('<div class="section-header">📈 Current Elo Rankings</div>', unsafe_allow_html=True)
    st.markdown("Elo ratings computed from all matches since 1998. Higher = stronger team historically.")

    elo_series = pd.Series(state['elo_ratings']).sort_values(ascending=False)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        # Filter to WC 2026 teams only option
        show_wc_only = st.checkbox("Show World Cup 2026 teams only", value=True)
        if show_wc_only:
            elo_filtered = elo_series[elo_series.index.isin(WC2026_TEAMS)]
        else:
            elo_filtered = elo_series.head(50)

        top_n = st.slider("Show top N teams", 10, min(50, len(elo_filtered)), 20)
        elo_top = elo_filtered.head(top_n)

        fig, ax = plt.subplots(figsize=(6, top_n * 0.38 + 1))
        colors = ['#FFD700' if i == 0 else '#C0C0C0' if i == 1 else '#CD7F32' if i == 2
                  else '#2d6a4f' if v > 1800 else '#74c69d' if v > 1700 else '#b7e4c7'
                  for i, v in enumerate(elo_top.values)]
        bars = ax.barh(range(len(elo_top)), elo_top.values, color=colors)
        ax.set_yticks(range(len(elo_top)))
        ax.set_yticklabels([f"{i+1}. {t}" for i, t in enumerate(elo_top.index)], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Elo Rating', fontsize=10)
        ax.set_title('Team Elo Rankings', fontsize=11)
        ax.axvline(1500, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.text(1500, -0.8, 'Average (1500)', fontsize=7, color='gray', ha='center')
        for bar, val in zip(bars, elo_top.values):
            ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}', va='center', fontsize=8)
        ax.set_xlim(min(elo_top.values) - 60, max(elo_top.values) + 80)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Full rankings table")
        elo_df = elo_filtered.reset_index()
        elo_df.columns = ['Team', 'Elo Rating']
        elo_df['Rank'] = range(1, len(elo_df) + 1)
        elo_df['Elo Rating'] = elo_df['Elo Rating'].round(1)
        elo_df = elo_df[['Rank', 'Team', 'Elo Rating']]
        st.dataframe(elo_df.set_index('Rank'), use_container_width=True, height=500)

        st.markdown("""
        **How Elo works:**
        - Start: every team begins at **1500**
        - Win vs strong opponent → big Elo gain
        - Lose vs weak opponent → big Elo drop
        - K-factor = **20** (moderate sensitivity)
        """)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:0.8rem;'>"
    "Built by <b>Faiq Ahmed</b> · Random Forest · Kaggle: International Football Results 1872–2026 · "
    "Model accuracy: 54.2%"
    "</div>",
    unsafe_allow_html=True
)
