from pathlib import Path
from html import escape

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


APP_DIR = Path(__file__).resolve().parent
PREFERRED_DATA_FILE = APP_DIR / "fra_cleaned_analysis_ready.csv"
FALLBACK_DATA_FILE = APP_DIR / "fra_cleaned.csv"

SCENT_COLUMNS = [
    "mainaccord1",
    "mainaccord2",
    "mainaccord3",
    "mainaccord4",
    "mainaccord5",
    "Top",
    "Middle",
    "Base",
]

RESULT_COLUMNS = [
    "Perfume",
    "Brand",
    "Country",
    "Gender",
    "Rating Value",
    "Rating Count",
    "Popularity Score",
    "Similarity",
    "Year",
]

SIMILAR_HISTORY_COLUMNS = [
    "Perfume",
    "Brand",
    "Country",
    "Gender",
    "Rating Value",
    "Rating Count",
    "Popularity Score",
    "High_Potential",
    "Similarity",
    "Year",
]


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ivory: #fbf8f1;
            --champagne: #eadcc5;
            --soft-gold: #b48a45;
            --warm-brown: #6e5a44;
            --ink: #2f2f2f;
            --muted: #706b64;
            --line: #e6ddd0;
            --card: rgba(255, 253, 248, 0.94);
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(228, 211, 184, 0.32), transparent 34rem),
                linear-gradient(180deg, #fbf8f1 0%, #f7f3eb 42%, #f4f2ee 100%);
            color: var(--ink);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2rem 2.1rem;
            margin-bottom: 1.35rem;
            border: 1px solid rgba(180, 138, 69, 0.22);
            border-radius: 18px;
            background: linear-gradient(135deg, #fffdf7 0%, #f1e3ca 100%);
            box-shadow: 0 18px 48px rgba(82, 64, 42, 0.10);
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 13rem;
            height: 13rem;
            right: -4.5rem;
            top: -5rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.42);
            border: 1px solid rgba(180, 138, 69, 0.18);
        }

        .hero-kicker {
            color: var(--soft-gold);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .hero h1 {
            color: #282521;
            font-size: clamp(2rem, 4vw, 3.3rem);
            line-height: 1.05;
            margin: 0 0 0.75rem 0;
            letter-spacing: 0;
        }

        .hero-subtitle {
            color: #564b3d;
            font-size: 1.08rem;
            margin: 0 0 0.55rem 0;
            max-width: 820px;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 0.98rem;
            max-width: 820px;
            margin: 0;
        }

        .section-lead, .info-box {
            padding: 1rem 1.15rem;
            margin: 0.7rem 0 1rem 0;
            border-radius: 14px;
            border: 1px solid var(--line);
            background: rgba(255, 253, 248, 0.76);
            box-shadow: 0 10px 26px rgba(82, 64, 42, 0.06);
            color: #4d4944;
        }

        .metric-card, .perfume-card, .target-card {
            padding: 1rem 1.05rem;
            border-radius: 14px;
            border: 1px solid var(--line);
            background: var(--card);
            box-shadow: 0 12px 30px rgba(82, 64, 42, 0.08);
            height: 100%;
        }

        .metric-label, .card-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            color: #2f2a24;
            font-size: 1.65rem;
            font-weight: 750;
            line-height: 1.1;
        }

        .perfume-card-title, .target-title {
            color: #282521;
            font-size: 1.02rem;
            font-weight: 700;
            line-height: 1.25;
            margin-bottom: 0.45rem;
        }

        .card-row {
            color: #59524a;
            font-size: 0.9rem;
            margin-top: 0.28rem;
        }

        .tag {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            margin: 0.15rem 0.2rem 0.15rem 0;
            border-radius: 999px;
            border: 1px solid rgba(180, 138, 69, 0.32);
            background: rgba(234, 220, 197, 0.35);
            color: #6a563e;
            font-size: 0.78rem;
            font-weight: 650;
        }

        .divider {
            height: 1px;
            margin: 1.15rem 0;
            background: linear-gradient(90deg, transparent, rgba(180, 138, 69, 0.35), transparent);
        }

        div.stButton > button,
        div.stFormSubmitButton > button {
            border: 1px solid #9f7b42;
            background: #3a342e;
            color: #fffaf0;
            border-radius: 999px;
            padding: 0.58rem 1.2rem;
            box-shadow: 0 8px 20px rgba(58, 52, 46, 0.16);
        }

        div.stButton > button:hover,
        div.stFormSubmitButton > button:hover {
            border-color: #b48a45;
            background: #4a4138;
            color: #fffaf0;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            border-bottom: 1px solid var(--line);
        }

        .stTabs [data-baseweb="tab"] {
            color: #5f574f;
            border-radius: 999px 999px 0 0;
            padding: 0.7rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize common text placeholders for missing values."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    text_columns = df.select_dtypes(include=["object"]).columns
    for col in text_columns:
        df[col] = df[col].astype("string").str.strip()

    df = df.replace(
        {
            "": pd.NA,
            "unknown": pd.NA,
            "Unknown": pd.NA,
            "nan": pd.NA,
        }
    )
    return df


def to_numeric_column(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def make_scent_text(df: pd.DataFrame) -> pd.Series:
    existing_columns = [col for col in SCENT_COLUMNS if col in df.columns]
    if not existing_columns:
        return pd.Series([""] * len(df), index=df.index)

    return (
        df[existing_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore")


@st.cache_data
def load_data() -> pd.DataFrame:
    if PREFERRED_DATA_FILE.exists():
        data_path = PREFERRED_DATA_FILE
        try:
            df = pd.read_csv(data_path)
        except UnicodeDecodeError:
            df = pd.read_csv(data_path, encoding="cp1252")
    elif FALLBACK_DATA_FILE.exists():
        data_path = FALLBACK_DATA_FILE
        df = pd.read_csv(
            data_path,
            sep=";",
            encoding="cp1252",
            decimal=",",
        )
    else:
        raise FileNotFoundError(
            "请将 fra_cleaned_analysis_ready.csv 或 fra_cleaned.csv 放在 app.py 所在文件夹。"
        )

    df = normalize_missing_values(df)

    for col in ["Rating Value", "Rating Count", "Year"]:
        to_numeric_column(df, col)

    if "Rating Value" not in df.columns:
        df["Rating Value"] = np.nan
    if "Rating Count" not in df.columns:
        df["Rating Count"] = np.nan

    df["Popularity Score"] = df["Rating Value"].fillna(0) * np.log1p(
        df["Rating Count"].fillna(0)
    )

    threshold = df["Popularity Score"].quantile(0.9)
    if pd.isna(threshold):
        threshold = 0
    df["High_Potential"] = (df["Popularity Score"] >= threshold).astype(int)

    df["feature_text"] = make_scent_text(df)
    df["scent_text"] = df["feature_text"]

    for col in ["Perfume", "Brand", "Country", "Gender", "Year"]:
        if col not in df.columns:
            df[col] = pd.NA

    return df.reset_index(drop=True)


@st.cache_resource
def build_recommendation_model(feature_text_values: tuple[str, ...]):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=3000,
        ngram_range=(1, 2),
    )

    clean_features = [text if text.strip() else "missing_feature" for text in feature_text_values]
    tfidf_matrix = vectorizer.fit_transform(clean_features)
    return vectorizer, tfidf_matrix


@st.cache_resource
def build_potential_model(model_data: pd.DataFrame):
    training = model_data.copy()

    for col in ["Brand", "Country", "Gender", "Year", "scent_text"]:
        if col not in training.columns:
            training[col] = pd.NA

    training["Brand"] = training["Brand"].fillna("N/A").astype(str)
    training["Country"] = training["Country"].fillna("N/A").astype(str)
    training["Gender"] = training["Gender"].fillna("N/A").astype(str)
    training["Year"] = pd.to_numeric(training["Year"], errors="coerce")
    training["scent_text"] = training["scent_text"].fillna("").astype(str)

    x = training[["Brand", "Country", "Gender", "Year", "scent_text"]]
    y = training["High_Potential"].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="N/A")),
                        ("onehot", one_hot_encoder()),
                    ]
                ),
                ["Brand", "Country", "Gender"],
            ),
            (
                "year",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                ["Year"],
            ),
            (
                "scent",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=1200,
                    ngram_range=(1, 2),
                    min_df=5,
                ),
                "scent_text",
            ),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        max_iter=600,
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
    )

    potential_pipeline = Pipeline(
        steps=[
            ("features", preprocessor),
            ("classifier", classifier),
        ]
    )
    potential_pipeline.fit(x, y)

    clean_scent = [
        text if text.strip() else "missing_feature"
        for text in training["scent_text"].fillna("").astype(str)
    ]
    scent_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1200,
        ngram_range=(1, 2),
        min_df=5,
    )
    scent_matrix = scent_vectorizer.fit_transform(clean_scent)
    high_potential_threshold = float(training["Popularity Score"].quantile(0.9))

    return potential_pipeline, scent_vectorizer, scent_matrix, high_potential_threshold


def display_name(row: pd.Series) -> str:
    perfume = row.get("Perfume")
    brand = row.get("Brand")
    year = row.get("Year")

    perfume_text = "N/A" if pd.isna(perfume) else str(perfume)
    brand_text = "N/A" if pd.isna(brand) else str(brand)
    year_text = "" if pd.isna(year) else f" ({int(year)})"
    return f"{perfume_text} - {brand_text}{year_text}"


def safe_value(value, default: str = "N/A") -> str:
    if pd.isna(value):
        return default
    return escape(str(value))


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">🧴 Fragrance Intelligence</div>
            <h1>Perfume Recommendation & Market Potential App</h1>
            <p class="hero-subtitle">
                A data-driven fragrance recommendation and new product potential evaluation tool
            </p>
            <p class="hero-copy">
                Based on Fragrantica perfume accords, notes, ratings, and historical popularity patterns.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_target_card(row: pd.Series) -> None:
    rating = safe_value(row.get("Rating Value"))
    count = safe_value(row.get("Rating Count"))
    country = safe_value(row.get("Country"))
    gender = safe_value(row.get("Gender"))
    year = safe_value(row.get("Year"))
    popularity = row.get("Popularity Score")
    popularity_text = "N/A" if pd.isna(popularity) else f"{float(popularity):.3f}"

    st.markdown(
        f"""
        <div class="target-card">
            <div class="card-label">Selected Perfume</div>
            <div class="target-title">{safe_value(row.get("Perfume"))}</div>
            <span class="tag">{safe_value(row.get("Brand"))}</span>
            <span class="tag">{gender}</span>
            <span class="tag">{country}</span>
            <span class="tag">{year}</span>
            <div class="divider"></div>
            <div class="card-row">Rating Value: <b>{rating}</b></div>
            <div class="card-row">Rating Count: <b>{count}</b></div>
            <div class="card-row">Popularity Score: <b>{popularity_text}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_perfume_card(row: pd.Series, rank: int) -> None:
    st.markdown(
        f"""
        <div class="perfume-card">
            <div class="card-label">Top {rank} Match</div>
            <div class="perfume-card-title">{safe_value(row.get("Perfume"))}</div>
            <span class="tag">{safe_value(row.get("Brand"))}</span>
            <span class="tag">{safe_value(row.get("Gender"))}</span>
            <div class="divider"></div>
            <div class="card-row">Rating Value: <b>{safe_value(row.get("Rating Value"))}</b></div>
            <div class="card-row">Rating Count: <b>{safe_value(row.get("Rating Count"))}</b></div>
            <div class="card-row">Similarity: <b>{safe_value(row.get("Similarity"))}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def search_candidates(df: pd.DataFrame, keyword: str, min_rating_count: int) -> pd.DataFrame:
    filtered = df[df["Rating Count"].fillna(0) >= min_rating_count].copy()

    if not keyword.strip():
        return filtered.sort_values("Popularity Score", ascending=False).head(50)

    keyword = keyword.strip()
    perfume_match = filtered["Perfume"].fillna("").str.contains(keyword, case=False, na=False)
    brand_match = filtered["Brand"].fillna("").str.contains(keyword, case=False, na=False)
    return filtered[perfume_match | brand_match].sort_values(
        ["Popularity Score", "Rating Count"],
        ascending=[False, False],
    )


def recommend(
    df: pd.DataFrame,
    target_index: int,
    tfidf_matrix,
    top_n: int,
) -> pd.DataFrame:
    similarities = cosine_similarity(tfidf_matrix[target_index], tfidf_matrix).ravel()
    results = df.copy()
    results["Similarity"] = similarities
    results = results[results.index != target_index]
    results = results.sort_values(
        ["Similarity", "Popularity Score"],
        ascending=[False, False],
    ).head(top_n)

    for col in RESULT_COLUMNS:
        if col not in results.columns:
            results[col] = pd.NA

    display_results = results[RESULT_COLUMNS].copy()
    display_results["Popularity Score"] = display_results["Popularity Score"].round(3)
    display_results["Similarity"] = display_results["Similarity"].round(3)
    return display_results.fillna("N/A")


def make_new_perfume_input(
    brand: str,
    country: str,
    gender: str,
    year: int,
    main_accords: str,
    top_notes: str,
    middle_notes: str,
    base_notes: str,
) -> pd.DataFrame:
    scent_text = " ".join(
        [main_accords, top_notes, middle_notes, base_notes]
    ).strip()
    return pd.DataFrame(
        [
            {
                "Brand": brand.strip() or "N/A",
                "Country": country.strip() or "N/A",
                "Gender": gender,
                "Year": year,
                "scent_text": scent_text or "missing_feature",
            }
        ]
    )


def potential_level(probability: float) -> str:
    if probability >= 0.7:
        return "High Potential"
    if probability >= 0.4:
        return "Medium Potential"
    return "Low Potential"


def confidence_label(similarities: np.ndarray) -> tuple[str, int, float]:
    similar_count = int((similarities >= 0.20).sum())
    top5_average = float(np.mean(np.sort(similarities)[-5:])) if len(similarities) else 0

    if similar_count >= 50 and top5_average >= 0.35:
        return "High", similar_count, top5_average
    if similar_count >= 15 and top5_average >= 0.25:
        return "Medium", similar_count, top5_average
    return "Low", similar_count, top5_average


def similar_history_cases(
    df: pd.DataFrame,
    scent_vectorizer: TfidfVectorizer,
    scent_matrix,
    scent_text: str,
) -> tuple[pd.DataFrame, str, int, float]:
    query_text = scent_text if scent_text.strip() else "missing_feature"
    query_vector = scent_vectorizer.transform([query_text])
    similarities = cosine_similarity(query_vector, scent_matrix).ravel()

    confidence, similar_count, top5_average = confidence_label(similarities)
    top_indices = np.argsort(similarities)[::-1][:5]

    results = df.iloc[top_indices].copy()
    results["Similarity"] = similarities[top_indices]

    for col in SIMILAR_HISTORY_COLUMNS:
        if col not in results.columns:
            results[col] = pd.NA

    display_results = results[SIMILAR_HISTORY_COLUMNS].copy()
    display_results["Popularity Score"] = display_results["Popularity Score"].round(3)
    display_results["Similarity"] = display_results["Similarity"].round(3)
    return display_results.fillna("N/A"), confidence, similar_count, top5_average


def render_recommendation_page(data: pd.DataFrame, tfidf_matrix) -> None:
    st.sidebar.header("Search Settings")
    keyword = st.sidebar.text_input("香水名或品牌关键词")
    top_n = st.sidebar.slider("推荐数量", min_value=5, max_value=20, value=10)
    min_rating_count = st.sidebar.slider("最低评分人数", min_value=0, max_value=1000, value=50)

    candidates = search_candidates(data, keyword, min_rating_count)

    if keyword.strip():
        st.subheader("匹配到的候选香水")
    else:
        st.subheader("热门候选香水")

    if candidates.empty:
        st.info("没有找到匹配的香水。请尝试更换关键词，或降低最低评分人数。")
        return

    candidate_preview_columns = [
        col
        for col in ["Perfume", "Brand", "Country", "Gender", "Rating Value", "Rating Count", "Year"]
        if col in candidates.columns
    ]
    st.dataframe(
        candidates[candidate_preview_columns].fillna("N/A").head(50),
        use_container_width=True,
    )

    candidate_options = candidates.index.tolist()
    selected_index = st.selectbox(
        "选择目标香水",
        options=candidate_options,
        format_func=lambda idx: display_name(data.loc[idx]),
    )

    if st.button("生成推荐"):
        recommendations = recommend(data, selected_index, tfidf_matrix, top_n)
        st.subheader("推荐结果")
        st.dataframe(recommendations, use_container_width=True)


def render_potential_page(data: pd.DataFrame) -> None:
    st.subheader("New Perfume Potential Evaluator")
    st.write(
        "评估新品概念是否更接近历史高热度香水特征，不代表真实销量，也不代表一定好闻。"
    )

    try:
        model, scent_vectorizer, scent_matrix, threshold = build_potential_model(data)
    except Exception as exc:
        st.error(f"训练新品潜力模型时出现问题：{exc}")
        return

    with st.form("new_perfume_form"):
        brand = st.text_input("Brand", value="Dior")
        country = st.text_input("Country", value="France")
        gender = st.selectbox("Gender", options=["women", "men", "unisex"], index=0)
        year = st.number_input("Year", min_value=1900, max_value=2100, value=2026, step=1)
        main_accords = st.text_input(
            "Main accords",
            value="vanilla, amber, sweet, warm spicy, white floral",
        )
        top_notes = st.text_input("Top notes", value="orange blossom, pear, bergamot")
        middle_notes = st.text_input("Middle notes", value="coffee, jasmine, almond")
        base_notes = st.text_input("Base notes", value="vanilla, patchouli, cedar")
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    new_perfume = make_new_perfume_input(
        brand,
        country,
        gender,
        int(year),
        main_accords,
        top_notes,
        middle_notes,
        base_notes,
    )

    probability_values = model.predict_proba(new_perfume)[0]
    positive_class_index = list(model.named_steps["classifier"].classes_).index(1)
    probability = float(probability_values[positive_class_index])
    level = potential_level(probability)

    similar_cases, confidence, similar_count, top5_average = similar_history_cases(
        data,
        scent_vectorizer,
        scent_matrix,
        new_perfume.loc[0, "scent_text"],
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted High Potential Probability", f"{probability:.1%}")
    col2.metric("Potential Level", level)
    col3.metric("Prediction Confidence", confidence)

    st.write(f"High Potential Threshold: {threshold:.3f}")
    st.write(
        f"Similarity support: {similar_count} historical perfumes with similarity >= 0.20; "
        f"Top 5 average similarity = {top5_average:.3f}."
    )
    st.info(
        "这是基于历史 Fragrantica 数据的市场潜力参考，衡量新品概念是否接近历史高热度香水特征；"
        "它不代表真实销量，也不代表香水一定好闻。"
    )

    st.subheader("Top 5 Similar Historical Perfumes")
    st.dataframe(similar_cases, use_container_width=True)


def render_recommendation_page(data: pd.DataFrame, tfidf_matrix) -> None:
    st.sidebar.header("Search Settings")
    keyword = st.sidebar.text_input("Perfume or brand keyword")
    top_n = st.sidebar.slider("Number of recommendations", min_value=5, max_value=20, value=10)
    min_rating_count = st.sidebar.slider("Minimum rating count", min_value=0, max_value=1000, value=50)

    st.subheader("Find Similar Perfumes")
    st.markdown(
        """
        <div class="section-lead">
            🌸 Search by perfume or brand, select one candidate, then generate perfumes with similar
            accord and note profiles. Results are ranked by scent-text similarity, with popularity used
            as a secondary signal.
        </div>
        """,
        unsafe_allow_html=True,
    )

    candidates = search_candidates(data, keyword, min_rating_count)

    if keyword.strip():
        st.markdown("#### Matched Candidate Perfumes")
    else:
        st.markdown("#### Popular Candidate Perfumes")

    if candidates.empty:
        st.info("No matching perfume was found. Try another keyword or lower the minimum rating count.")
        return

    candidate_preview_columns = [
        col
        for col in ["Perfume", "Brand", "Country", "Gender", "Rating Value", "Rating Count", "Year"]
        if col in candidates.columns
    ]
    st.dataframe(
        candidates[candidate_preview_columns].fillna("N/A").head(50),
        use_container_width=True,
    )

    candidate_options = candidates.index.tolist()
    selected_index = st.selectbox(
        "Select target perfume",
        options=candidate_options,
        format_func=lambda idx: display_name(data.loc[idx]),
    )

    render_target_card(data.loc[selected_index])

    if st.button("Generate Recommendations"):
        recommendations = recommend(data, selected_index, tfidf_matrix, top_n)

        st.markdown("#### Top 3 Recommended Perfumes")
        top_rows = list(recommendations.head(3).iterrows())
        top_columns = st.columns(len(top_rows)) if top_rows else []
        for rank, (card_col, (_, row)) in enumerate(zip(top_columns, top_rows), start=1):
            with card_col:
                render_perfume_card(row, rank)

        st.markdown(
            """
            <div class="info-box">
                ✨ The table below keeps the full recommendation list for comparison, including rating
                context, popularity score, and cosine similarity.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("Recommendation Results")
        st.dataframe(recommendations, use_container_width=True)


def render_potential_page(data: pd.DataFrame) -> None:
    st.subheader("New Perfume Potential Evaluator")
    st.markdown(
        """
        <div class="section-lead">
            🧴 Enter a new perfume concept to estimate whether its pre-launch attributes resemble
            historically high-popularity perfumes in the dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        model, scent_vectorizer, scent_matrix, threshold = build_potential_model(data)
    except Exception as exc:
        st.error(f"Training the new perfume potential model failed: {exc}")
        return

    with st.form("new_perfume_form"):
        brand = st.text_input("Brand", value="Dior")
        country = st.text_input("Country", value="France")
        gender = st.selectbox("Gender", options=["women", "men", "unisex"], index=0)
        year = st.number_input("Year", min_value=1900, max_value=2100, value=2026, step=1)
        main_accords = st.text_input(
            "Main accords",
            value="vanilla, amber, sweet, warm spicy, white floral",
        )
        top_notes = st.text_input("Top notes", value="orange blossom, pear, bergamot")
        middle_notes = st.text_input("Middle notes", value="coffee, jasmine, almond")
        base_notes = st.text_input("Base notes", value="vanilla, patchouli, cedar")
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    new_perfume = make_new_perfume_input(
        brand,
        country,
        gender,
        int(year),
        main_accords,
        top_notes,
        middle_notes,
        base_notes,
    )

    probability_values = model.predict_proba(new_perfume)[0]
    positive_class_index = list(model.named_steps["classifier"].classes_).index(1)
    probability = float(probability_values[positive_class_index])
    level = potential_level(probability)

    similar_cases, confidence, similar_count, top5_average = similar_history_cases(
        data,
        scent_vectorizer,
        scent_matrix,
        new_perfume.loc[0, "scent_text"],
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("High Potential Probability", f"{probability:.1%}")
    with col2:
        render_metric_card("Potential Level", level)
    with col3:
        render_metric_card("Prediction Confidence", confidence)

    st.markdown(
        f"""
        <div class="info-box">
            <b>High Potential Threshold:</b> {threshold:.3f}<br>
            <b>Similarity support:</b> {similar_count} historical perfumes with similarity >= 0.20;
            Top 5 average similarity = {top5_average:.3f}.
            <div class="divider"></div>
            This model evaluates whether the new perfume concept is similar to historically
            high-popularity perfumes in the Fragrantica dataset. It does not predict real sales
            or guarantee that the perfume will smell good.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Top 5 Similar Historical Perfumes")
    st.markdown(
        """
        <div class="info-box">
            Similar historical perfumes are provided to avoid interpreting the prediction as a black-box result.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(similar_cases, use_container_width=True)


st.set_page_config(page_title="Perfume Recommendation App", layout="wide")
inject_custom_css()
render_hero()

try:
    data = load_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"读取或清洗数据时出现问题：{exc}")
    st.stop()

_, recommendation_tfidf = build_recommendation_model(
    tuple(data["feature_text"].fillna("").astype(str))
)

recommendation_tab, potential_tab = st.tabs(
    ["Find Similar Perfumes", "New Perfume Potential Evaluator"]
)

with recommendation_tab:
    render_recommendation_page(data, recommendation_tfidf)

with potential_tab:
    render_potential_page(data)
