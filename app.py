import streamlit as st
import pandas as pd
import plotly.express as px

from PIL import Image

from src.predict import predict_text
from src.ocr import extract_text_and_comments_from_image


# ==================================================
# PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="AI Hate Speech Moderation",
    page_icon="🛡️",
    layout="wide"
)


# ==================================================
# PAGE DESIGN
# ==================================================

st.markdown(
    """
    <style>

    /* MAIN BACKGROUND */

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(59, 130, 246, 0.15),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 85%,
                rgba(139, 92, 246, 0.12),
                transparent 28%
            ),
            linear-gradient(
                135deg,
                #f8fbff 0%,
                #eef4ff 50%,
                #f8faff 100%
            );

        color: #0f172a;
    }


    [data-testid="stHeader"] {
        background: transparent;
    }


    .block-container {
        max-width: 1080px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }


    h1, h2, h3, h4, p, label {
        color: #0f172a !important;
    }


    [data-testid="stCaptionContainer"] {
        color: #64748b !important;
    }


    /* BANNER */

    [data-testid="stImage"] img {
        border-radius: 18px;

        box-shadow:
            0 10px 30px
            rgba(30, 64, 175, 0.10);
    }


    /* TEXT AREA */

    textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;

        border-radius: 14px !important;

        border:
            1px solid #cbd5e1 !important;

        box-shadow:
            0 5px 18px
            rgba(15, 23, 42, 0.05);
    }


    textarea::placeholder {
        color: #64748b !important;
    }


    /* BUTTONS */

    div.stButton > button {
        width: 100%;
        min-height: 48px;

        border-radius: 12px;

        border: none;

        background:
            linear-gradient(
                90deg,
                #2563eb,
                #4f46e5
            );

        color: white !important;

        font-size: 16px;
        font-weight: 700;

        box-shadow:
            0 7px 18px
            rgba(37, 99, 235, 0.20);
    }


    div.stButton > button:hover {

        background:
            linear-gradient(
                90deg,
                #1d4ed8,
                #4338ca
            );

        color: white !important;

        transform: translateY(-1px);
    }


    /* RESULT CONTAINERS */

    [data-testid="stVerticalBlockBorderWrapper"] {

        background:
            rgba(255, 255, 255, 0.80);

        border-radius: 14px;

        box-shadow:
            0 5px 16px
            rgba(15, 23, 42, 0.04);
    }


    /* EXPANDERS */

    [data-testid="stExpander"] {

        background:
            rgba(255, 255, 255, 0.68);

        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# SESSION STATE
# ==================================================

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []


if "last_result" not in st.session_state:
    st.session_state.last_result = None


if "last_text" not in st.session_state:
    st.session_state.last_text = ""


if "last_explanation" not in st.session_state:
    st.session_state.last_explanation = None


if "batch_results" not in st.session_state:
    st.session_state.batch_results = None


if "ocr_extracted_text" not in st.session_state:
    st.session_state.ocr_extracted_text = None


if "detected_comments" not in st.session_state:
    st.session_state.detected_comments = []


if "image_results" not in st.session_state:
    st.session_state.image_results = None


# ==================================================
# SAVE RESULT TO DASHBOARD HISTORY
# ==================================================

def save_to_history(text, result, source):

    st.session_state.analysis_history.append(
        {
            "Text": text,

            "Prediction":
                result["label"],

            "Confidence":
                round(
                    result["confidence"] * 100,
                    2
                ),

            "Action":
                result["action"],

            "Source":
                source
        }
    )


# ==================================================
# EXPLAINABLE AI
# ==================================================

def explain_prediction(
    text,
    original_result,
    max_words=15
):

    """
    Lightweight Explainable AI.

    Removes meaningful words individually and checks
    how much the prediction changes.

    Common words such as 'you', 'the', 'is', etc.
    are ignored.
    """

    words = text.split()


    if len(words) <= 1:
        return pd.DataFrame()


    # Common words we do not want to display
    # as harmful/influential words

    stop_words = {

        "i", "me", "my", "mine",

        "you", "your", "yours",

        "he", "him", "his",

        "she", "her", "hers",

        "we", "us", "our", "ours",

        "they", "them",
        "their", "theirs",

        "a", "an", "the",

        "is", "am", "are",
        "was", "were",

        "be", "been", "being",

        "do", "does", "did",

        "have", "has", "had",

        "this", "that",
        "these", "those",

        "and", "or", "but",

        "to", "of", "in", "on",
        "at", "for", "from",
        "with",

        "it", "its",

        "very", "really",
        "just", "such", "so"
    }


    original_label = (
        original_result["label"]
    )


    original_confidence = (
        original_result["confidence"]
    )


    # Limit number of model runs on CPU

    if len(words) > max_words:

        step = max(
            1,
            len(words) // max_words
        )


        word_indexes = list(
            range(
                0,
                len(words),
                step
            )
        )[:max_words]


    else:

        word_indexes = list(
            range(len(words))
        )


    importance_scores = []


    for index in word_indexes:

        clean_word = (
            words[index]
            .strip(
                ".,!?;:\"'()[]{}"
            )
            .lower()
        )


        # Ignore common words

        if (
            not clean_word
            or clean_word in stop_words
            or len(clean_word) <= 2
        ):

            continue


        modified_words = (
            words[:index]
            +
            words[index + 1:]
        )


        modified_text = " ".join(
            modified_words
        )


        if not modified_text.strip():

            continue


        changed_result = predict_text(
            modified_text
        )


        # Same prediction:
        # measure confidence drop

        if (
            changed_result["label"]
            == original_label
        ):

            score = (
                original_confidence
                -
                changed_result[
                    "confidence"
                ]
            )


        # Prediction changed:
        # strong influence

        else:

            score = (
                original_confidence
            )


        score = max(
            score,
            0
        )


        if score > 0:

            importance_scores.append(
                {
                    "Word":
                        clean_word,

                    "Influence":
                        score
                }
            )


    if not importance_scores:

        return pd.DataFrame()


    explanation_df = pd.DataFrame(
        importance_scores
    )


    # Merge repeated words

    explanation_df = (
        explanation_df
        .groupby(
            "Word",
            as_index=False
        )["Influence"]
        .max()
    )


    explanation_df = (
        explanation_df[
            explanation_df[
                "Influence"
            ] > 0
        ]
    )


    if explanation_df.empty:

        return explanation_df


    maximum_score = (
        explanation_df[
            "Influence"
        ].max()
    )


    if maximum_score > 0:

        explanation_df[
            "Influence"
        ] = (

            explanation_df[
                "Influence"
            ]

            /

            maximum_score

        ) * 100


    explanation_df = (
        explanation_df
        .sort_values(
            "Influence",
            ascending=False
        )
        .head(8)
    )


    return explanation_df


# ==================================================
# RESULT CARDS
# ==================================================

def show_result_cards(result):

    label = result["label"]

    confidence = (
        result["confidence"] * 100
    )

    action = result["action"]

    message = result["message"]


    col1, col2, col3 = (
        st.columns(3)
    )


    # ------------------------------------------------
    # PREDICTION
    # ------------------------------------------------

    with col1:

        with st.container(
            border=True
        ):

            st.caption(
                "🔎 PREDICTION"
            )


            if label == "Hate Speech":

                st.error(
                    f"### {label}"
                )


            elif (
                label
                == "Offensive Language"
            ):

                st.warning(
                    f"### {label}"
                )


            else:

                st.success(
                    f"### {label}"
                )


    # ------------------------------------------------
    # CONFIDENCE
    # ------------------------------------------------

    with col2:

        with st.container(
            border=True
        ):

            st.caption(
                "📊 MODEL CONFIDENCE"
            )


            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%",
                label_visibility="collapsed"
            )


            st.progress(
                min(
                    int(confidence),
                    100
                )
            )


    # ------------------------------------------------
    # ACTION
    # ------------------------------------------------

    with col3:

        with st.container(
            border=True
        ):

            st.caption(
                "🛡️ MODERATION ACTION"
            )


            if action == "BLOCK":

                st.error(
                    "### 🚫 BLOCK"
                )


            elif action == "WARN":

                st.warning(
                    "### ⚠️ WARN"
                )


            elif action == "REVIEW":

                st.info(
                    "### 👤 REVIEW"
                )


            else:

                st.success(
                    "### ✅ ALLOW"
                )


    # ------------------------------------------------
    # EXPLANATION
    # ------------------------------------------------

    st.write("")


    with st.container(
        border=True
    ):

        st.subheader(
            "💡 Why this decision?"
        )


        st.write(
            message
        )


# ==================================================
# HEADER BANNER
# ==================================================

st.image(
    "assets/header_banner.png",
    use_container_width=True
)


# ==================================================
# INTRODUCTION
# ==================================================

st.markdown(
    "### 🛡️ Hate Speech & Offensive Language Detection using DistilRoBERTa"
)


st.caption(
    "AI-Powered Social Media Content Moderation"
)


st.write(
    "Analyze social-media posts, comments, tweets, "
    "messages, files or screenshots and receive "
    "an AI-assisted moderation recommendation."
)


st.divider()


# ==================================================
# MAIN TABS
# ==================================================

single_tab, batch_tab, image_tab, dashboard_tab = (
    st.tabs(
        [
            "🔎 Single Text Analysis",
            "📁 Batch File Analysis",
            "🖼️ Image Text Analysis",
            "📊 Dashboard"
        ]
    )
)


# ==================================================
# TAB 1 - SINGLE TEXT ANALYSIS
# ==================================================

with single_tab:

    st.subheader(
        "📝 Analyze Social Media Content"
    )


    st.caption(
        "Enter or paste the content "
        "you want to analyze."
    )


    text = st.text_area(
        "Social Media Content",

        placeholder=(
            "Type a post, tweet, "
            "comment or message here..."
        ),

        height=150,

        label_visibility="collapsed",

        key="single_text"
    )


    analyze_button = st.button(
        "🔍 Analyze Content",

        type="primary",

        use_container_width=True,

        key="single_analyze_button"
    )


    # ==============================================
    # RUN SINGLE PREDICTION
    # ==============================================

    if analyze_button:

        if not text.strip():

            st.warning(
                "⚠️ Please enter some "
                "content before analyzing."
            )


        else:

            with st.spinner(
                "DistilRoBERTa is analyzing "
                "the content..."
            ):

                result = predict_text(
                    text
                )


            st.session_state.last_result = (
                result
            )


            st.session_state.last_text = (
                text
            )


            st.session_state.last_explanation = (
                None
            )


            save_to_history(
                text,
                result,
                "Single"
            )


    # ==============================================
    # SHOW LAST SINGLE RESULT
    # ==============================================

    if (
        st.session_state.last_result
        is not None
    ):

        result = (
            st.session_state.last_result
        )


        confidence = (
            result["confidence"] * 100
        )


        st.divider()


        st.header(
            "📊 Analysis Result"
        )


        show_result_cards(
            result
        )


        # ==========================================
        # VISUAL REPRESENTATION
        # ==========================================

        st.write("")


        st.subheader(
            "📈 Visual Representation"
        )


        chart_col1, chart_col2 = (
            st.columns(2)
        )


        # ------------------------------------------
        # CONFIDENCE DONUT
        # ------------------------------------------

        with chart_col1:

            confidence_df = (
                pd.DataFrame(
                    {
                        "Category": [
                            "Prediction Confidence",
                            "Remaining Uncertainty"
                        ],

                        "Value": [
                            confidence,
                            100 - confidence
                        ]
                    }
                )
            )


            donut_fig = px.pie(
                confidence_df,

                names="Category",

                values="Value",

                hole=0.55,

                title=(
                    "Current Prediction "
                    "Confidence"
                )
            )


            donut_fig.update_traces(
                textinfo="percent+label"
            )


            st.plotly_chart(
                donut_fig,
                use_container_width=True
            )


        # ------------------------------------------
        # CURRENT CLASSIFICATION
        # ------------------------------------------

        with chart_col2:

            current_df = (
                pd.DataFrame(
                    {
                        "Result": [
                            result["label"]
                        ],

                        "Confidence": [
                            confidence
                        ]
                    }
                )
            )


            current_fig = px.bar(
                current_df,

                x="Result",

                y="Confidence",

                text="Confidence",

                title="Current Classification"
            )


            current_fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )


            current_fig.update_yaxes(
                range=[
                    0,
                    100
                ],

                title="Confidence (%)"
            )


            st.plotly_chart(
                current_fig,
                use_container_width=True
            )


        # ==========================================
        # EXPLAINABLE AI
        # ==========================================

        st.write("")


        st.subheader(
            "🧠 Explainable AI"
        )


        st.write(
            "Discover which meaningful words "
            "had the strongest influence on "
            "the AI's prediction."
        )


        explain_button = st.button(
            "✨ Explain This Prediction",

            use_container_width=True,

            key="explain_button"
        )


        if explain_button:

            with st.spinner(
                "Analyzing word influence..."
            ):

                explanation = (
                    explain_prediction(
                        st.session_state.last_text,
                        st.session_state.last_result
                    )
                )


            st.session_state.last_explanation = (
                explanation
            )


        if (
            st.session_state.last_explanation
            is not None
        ):

            explanation_df = (
                st.session_state.last_explanation
            )


            if explanation_df.empty:

                st.info(
                    "No individual meaningful word "
                    "strongly changed the model's prediction."
                )


            else:

                explanation_sorted = (
                    explanation_df
                    .sort_values(
                        "Influence",
                        ascending=True
                    )
                )


                word_fig = px.bar(
                    explanation_sorted,

                    x="Influence",

                    y="Word",

                    orientation="h",

                    text="Influence",

                    title=(
                        "Words Influencing "
                        "the Prediction"
                    )
                )


                word_fig.update_traces(
                    texttemplate="%{text:.1f}",
                    textposition="outside"
                )


                word_fig.update_xaxes(
                    title=(
                        "Relative Influence Score"
                    )
                )


                st.plotly_chart(
                    word_fig,
                    use_container_width=True
                )


                top_words = (
                    explanation_df[
                        "Word"
                    ]
                    .head(5)
                    .tolist()
                )


                st.write(
                    "**Most influential words:** "
                    + ", ".join(
                        top_words
                    )
                )


                st.caption(
                    "This uses a lightweight "
                    "word-removal explanation method. "
                    "Common words such as pronouns "
                    "and articles are filtered out."
                )


# ==================================================
# TAB 2 - BATCH FILE ANALYSIS
# ==================================================

with batch_tab:

    st.subheader(
        "📁 Batch Content Moderation"
    )


    st.write(
        "Upload a CSV or TXT file and analyze "
        "multiple social-media messages at once."
    )


    uploaded_file = st.file_uploader(
        "Upload CSV or TXT file",

        type=[
            "csv",
            "txt"
        ],

        key="batch_file"
    )


    batch_texts = []


    if uploaded_file is not None:

        file_name = (
            uploaded_file.name.lower()
        )


        # ==========================================
        # CSV
        # ==========================================

        if file_name.endswith(
            ".csv"
        ):

            try:

                input_df = pd.read_csv(
                    uploaded_file
                )


                if input_df.empty:

                    st.warning(
                        "The uploaded CSV "
                        "file is empty."
                    )


                else:

                    st.write(
                        "#### File Preview"
                    )


                    st.dataframe(
                        input_df.head(10),

                        use_container_width=True
                    )


                    column_names = (
                        input_df.columns.tolist()
                    )


                    preferred_names = [
                        "text",
                        "tweet",
                        "comment",
                        "content",
                        "message",
                        "post"
                    ]


                    lower_columns = [
                        str(c).lower()
                        for c
                        in column_names
                    ]


                    default_index = 0


                    for preferred in (
                        preferred_names
                    ):

                        if (
                            preferred
                            in lower_columns
                        ):

                            default_index = (
                                lower_columns
                                .index(
                                    preferred
                                )
                            )

                            break


                    text_column = (
                        st.selectbox(
                            "Select the column "
                            "containing text:",

                            column_names,

                            index=default_index
                        )
                    )


                    batch_texts = (
                        input_df[
                            text_column
                        ]
                        .dropna()
                        .astype(str)
                        .tolist()
                    )


            except Exception as error:

                st.error(
                    f"Could not read "
                    f"CSV file: {error}"
                )


        # ==========================================
        # TXT
        # ==========================================

        else:

            try:

                file_content = (
                    uploaded_file
                    .getvalue()
                    .decode(
                        "utf-8",
                        errors="ignore"
                    )
                )


                batch_texts = [
                    line.strip()

                    for line
                    in file_content.splitlines()

                    if line.strip()
                ]


                st.write(
                    f"**Detected "
                    f"{len(batch_texts)} "
                    f"text entries.**"
                )


                if batch_texts:

                    preview_df = (
                        pd.DataFrame(
                            {
                                "Text":
                                    batch_texts[:10]
                            }
                        )
                    )


                    st.dataframe(
                        preview_df,
                        use_container_width=True
                    )


            except Exception as error:

                st.error(
                    f"Could not read "
                    f"TXT file: {error}"
                )


        # ==========================================
        # ANALYZE BATCH
        # ==========================================

        if batch_texts:

            max_possible = min(
                len(batch_texts),
                200
            )


            rows_to_process = (
                st.number_input(
                    "Number of rows to analyze",

                    min_value=1,

                    max_value=max_possible,

                    value=min(
                        max_possible,
                        50
                    ),

                    step=1
                )
            )


            st.caption(
                "For faster CPU performance, "
                "a maximum of 200 rows can "
                "be processed at one time."
            )


            batch_button = st.button(
                "🚀 Analyze Batch",

                type="primary",

                use_container_width=True,

                key="batch_analyze_button"
            )


            if batch_button:

                selected_texts = (
                    batch_texts[
                        :int(
                            rows_to_process
                        )
                    ]
                )


                batch_output = []


                progress_bar = (
                    st.progress(0)
                )


                status_text = (
                    st.empty()
                )


                total_rows = len(
                    selected_texts
                )


                for index, current_text in (
                    enumerate(
                        selected_texts
                    )
                ):

                    status_text.write(
                        f"Analyzing "
                        f"{index + 1} of "
                        f"{total_rows}..."
                    )


                    result = predict_text(
                        current_text
                    )


                    batch_output.append(
                        {
                            "Text":
                                current_text,

                            "Prediction":
                                result["label"],

                            "Confidence (%)":
                                round(
                                    result[
                                        "confidence"
                                    ] * 100,
                                    2
                                ),

                            "Action":
                                result["action"],

                            "Explanation":
                                result["message"]
                        }
                    )


                    save_to_history(
                        current_text,
                        result,
                        "Batch"
                    )


                    progress_bar.progress(
                        int(
                            (
                                index + 1
                            )
                            /
                            total_rows
                            *
                            100
                        )
                    )


                status_text.success(
                    "✅ Batch analysis completed!"
                )


                st.session_state.batch_results = (
                    pd.DataFrame(
                        batch_output
                    )
                )


        # ==========================================
        # SHOW BATCH RESULTS
        # ==========================================

        if (
            st.session_state.batch_results
            is not None
        ):

            results_df = (
                st.session_state.batch_results
            )


            st.divider()


            st.subheader(
                "📊 Batch Analysis Results"
            )


            st.dataframe(
                results_df,

                use_container_width=True,

                hide_index=True
            )


            class_counts = (
                results_df[
                    "Prediction"
                ]
                .value_counts()
                .reset_index()
            )


            class_counts.columns = [
                "Prediction",
                "Count"
            ]


            batch_col1, batch_col2 = (
                st.columns(2)
            )


            with batch_col1:

                batch_pie = px.pie(
                    class_counts,

                    names="Prediction",

                    values="Count",

                    hole=0.4,

                    title=(
                        "Batch Classification "
                        "Distribution"
                    )
                )


                st.plotly_chart(
                    batch_pie,

                    use_container_width=True
                )


            with batch_col2:

                batch_bar = px.bar(
                    class_counts,

                    x="Prediction",

                    y="Count",

                    text="Count",

                    title=(
                        "Batch Prediction "
                        "Summary"
                    )
                )


                batch_bar.update_traces(
                    textposition="outside"
                )


                st.plotly_chart(
                    batch_bar,

                    use_container_width=True
                )


            csv_data = (
                results_df
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                )
            )


            st.download_button(
                label=(
                    "⬇️ Download "
                    "Moderation Results"
                ),

                data=csv_data,

                file_name=(
                    "moderation_results.csv"
                ),

                mime="text/csv",

                use_container_width=True
            )


# ==================================================
# TAB 3 - IMAGE TEXT ANALYSIS
# ==================================================

with image_tab:

    st.subheader(
        "🖼️ Image Text Moderation"
    )


    st.write(
        "Upload a social-media screenshot "
        "containing one or more comments. "
        "The system extracts the visible text "
        "and attempts to separate each comment "
        "before moderation."
    )


    st.caption(
        "Supported formats: "
        "PNG, JPG, JPEG and WEBP"
    )


    uploaded_image = st.file_uploader(
        "Upload social-media screenshot",

        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ],

        key="image_uploader"
    )


    if uploaded_image is not None:

        try:

            image = Image.open(
                uploaded_image
            )


            # ======================================
            # DISPLAY IMAGE
            # ======================================

            st.write(
                "#### 📷 Uploaded Image"
            )


            st.image(
                image,

                caption=(
                    "Image selected "
                    "for moderation"
                ),

                use_container_width=True
            )


            # ======================================
            # EXTRACT COMMENTS
            # ======================================

            extract_button = st.button(
                "🔤 Extract Comments From Image",

                use_container_width=True,

                key="extract_image_comments"
            )


            if extract_button:

                with st.spinner(
                    "Reading and separating "
                    "comments from the screenshot..."
                ):

                    (
                        extracted_text,
                        detected_comments
                    ) = (
                        extract_text_and_comments_from_image(
                            image
                        )
                    )


                st.session_state[
                    "ocr_extracted_text"
                ] = extracted_text


                st.session_state[
                    "detected_comments"
                ] = detected_comments


                # Clear previous image result
                st.session_state[
                    "image_results"
                ] = None


            # ======================================
            # OCR RAW TEXT
            # ======================================

            extracted_text = (
                st.session_state.get(
                    "ocr_extracted_text"
                )
            )


            detected_comments = (
                st.session_state.get(
                    "detected_comments",
                    []
                )
            )


            if extracted_text:

                st.success(
                    "✅ Text extraction completed."
                )


                with st.expander(
                    "🔤 View Raw OCR Text"
                ):

                    st.text(
                        extracted_text
                    )


                # ==================================
                # DETECTED COMMENTS
                # ==================================

                st.write(
                    "### 💬 Detected Comments"
                )


                if not detected_comments:

                    st.warning(
                        "No individual comments "
                        "could be separated automatically. "
                        "Try a clearer screenshot."
                    )


                else:

                    st.write(
                        f"**{len(detected_comments)} "
                        f"comments detected.**"
                    )


                    st.caption(
                        "Check the comments below. "
                        "You can correct OCR mistakes "
                        "before analysis."
                    )


                    editable_comments = []


                    for index, comment in (
                        enumerate(
                            detected_comments
                        )
                    ):

                        edited_comment = (
                            st.text_area(
                                f"Comment "
                                f"{index + 1}",

                                value=comment,

                                height=80,

                                key=(
                                    f"image_comment_"
                                    f"{index}"
                                )
                            )
                        )


                        if (
                            edited_comment
                            .strip()
                        ):

                            editable_comments.append(
                                edited_comment
                                .strip()
                            )


                    # ==============================
                    # ANALYZE ALL COMMENTS
                    # ==============================

                    image_analyze_button = (
                        st.button(
                            "🔍 Analyze All Comments",

                            type="primary",

                            use_container_width=True,

                            key=(
                                "analyze_all_"
                                "image_comments"
                            )
                        )
                    )


                    if image_analyze_button:

                        if not editable_comments:

                            st.warning(
                                "No comments are "
                                "available for analysis."
                            )


                        else:

                            image_output = []


                            progress = (
                                st.progress(0)
                            )


                            status = (
                                st.empty()
                            )


                            total = len(
                                editable_comments
                            )


                            for index, comment in (
                                enumerate(
                                    editable_comments
                                )
                            ):

                                status.write(
                                    f"Analyzing comment "
                                    f"{index + 1} "
                                    f"of {total}..."
                                )


                                result = (
                                    predict_text(
                                        comment
                                    )
                                )


                                image_output.append(
                                    {
                                        "Comment":
                                            comment,

                                        "Prediction":
                                            result[
                                                "label"
                                            ],

                                        "Confidence (%)":
                                            round(
                                                result[
                                                    "confidence"
                                                ] * 100,
                                                2
                                            ),

                                        "Action":
                                            result[
                                                "action"
                                            ],

                                        "Explanation":
                                            result[
                                                "message"
                                            ]
                                    }
                                )


                                save_to_history(
                                    comment,
                                    result,
                                    "Image"
                                )


                                progress.progress(
                                    int(
                                        (
                                            index + 1
                                        )
                                        /
                                        total
                                        *
                                        100
                                    )
                                )


                            status.success(
                                "✅ All detected "
                                "comments analyzed!"
                            )


                            st.session_state[
                                "image_results"
                            ] = (
                                pd.DataFrame(
                                    image_output
                                )
                            )


            # ======================================
            # IMAGE RESULTS
            # ======================================

            if (
                st.session_state.image_results
                is not None
            ):

                image_results_df = (
                    st.session_state.image_results
                )


                st.divider()


                st.header(
                    "📊 Individual Comment Results"
                )


                # ==================================
                # SUMMARY COUNTS
                # ==================================

                total_comments = len(
                    image_results_df
                )


                hate_count = int(
                    (
                        image_results_df[
                            "Prediction"
                        ]
                        == "Hate Speech"
                    ).sum()
                )


                offensive_count = int(
                    (
                        image_results_df[
                            "Prediction"
                        ]
                        == "Offensive Language"
                    ).sum()
                )


                safe_count = int(
                    (
                        image_results_df[
                            "Prediction"
                        ]
                        == "Neither"
                    ).sum()
                )


                summary1, summary2, summary3, summary4 = (
                    st.columns(4)
                )


                summary1.metric(
                    "💬 Total Comments",
                    total_comments
                )


                summary2.metric(
                    "🚫 Hate Speech",
                    hate_count
                )


                summary3.metric(
                    "⚠️ Offensive",
                    offensive_count
                )


                summary4.metric(
                    "✅ Safe",
                    safe_count
                )


                # ==================================
                # TABLE
                # ==================================

                st.write(
                    "### 📝 Comment-by-Comment Analysis"
                )


                st.dataframe(
                    image_results_df,

                    use_container_width=True,

                    hide_index=True
                )


                # ==================================
                # DETAILED RESULTS
                # ==================================

                st.write(
                    "### 🔎 Detailed Results"
                )


                for index, row in (
                    image_results_df
                    .iterrows()
                ):

                    with st.expander(
                        f"Comment {index + 1}: "
                        f"{row['Prediction']} "
                        f"({row['Confidence (%)']}%)"
                    ):

                        st.write(
                            f"**Comment:** "
                            f"{row['Comment']}"
                        )


                        st.write(
                            f"**Prediction:** "
                            f"{row['Prediction']}"
                        )


                        st.write(
                            f"**Confidence:** "
                            f"{row['Confidence (%)']}%"
                        )


                        st.write(
                            f"**Moderation Action:** "
                            f"{row['Action']}"
                        )


                        st.write(
                            f"**Reason:** "
                            f"{row['Explanation']}"
                        )


                # ==================================
                # IMAGE RESULT GRAPHS
                # ==================================

                st.write(
                    "### 📈 Screenshot Moderation Summary"
                )


                image_summary_df = (
                    image_results_df[
                        "Prediction"
                    ]
                    .value_counts()
                    .reindex(
                        [
                            "Hate Speech",
                            "Offensive Language",
                            "Neither"
                        ],
                        fill_value=0
                    )
                    .reset_index()
                )


                image_summary_df.columns = [
                    "Prediction",
                    "Count"
                ]


                img_chart1, img_chart2 = (
                    st.columns(2)
                )


                with img_chart1:

                    image_pie = px.pie(
                        image_summary_df,

                        names="Prediction",

                        values="Count",

                        hole=0.45,

                        title=(
                            "Comment Classification "
                            "Distribution"
                        )
                    )


                    st.plotly_chart(
                        image_pie,

                        use_container_width=True
                    )


                with img_chart2:

                    image_bar = px.bar(
                        image_summary_df,

                        x="Prediction",

                        y="Count",

                        text="Count",

                        title=(
                            "Number of Comments "
                            "by Category"
                        )
                    )


                    image_bar.update_traces(
                        textposition="outside"
                    )


                    st.plotly_chart(
                        image_bar,

                        use_container_width=True
                    )


                # ==================================
                # DOWNLOAD IMAGE REPORT
                # ==================================

                image_csv = (
                    image_results_df
                    .to_csv(
                        index=False
                    )
                    .encode(
                        "utf-8"
                    )
                )


                st.download_button(
                    "⬇️ Download Image "
                    "Moderation Results",

                    data=image_csv,

                    file_name=(
                        "image_comment_results.csv"
                    ),

                    mime="text/csv",

                    use_container_width=True
                )


        except Exception as error:

            st.error(
                "❌ The uploaded image "
                "could not be processed."
            )


            st.caption(
                f"Technical details: {error}"
            )


# ==================================================
# TAB 4 - DASHBOARD
# ==================================================

with dashboard_tab:

    st.subheader(
        "📊 Moderation Dashboard"
    )


    st.write(
        "This dashboard summarizes all content "
        "analyzed during the current session."
    )


    if not st.session_state.analysis_history:

        st.info(
            "No content has been analyzed yet. "
            "Use Single Text, Batch File or "
            "Image Text Analysis first."
        )


    else:

        history_df = (
            pd.DataFrame(
                st.session_state.analysis_history
            )
        )


        total_analyzed = len(
            history_df
        )


        hate_count = int(
            (
                history_df[
                    "Prediction"
                ]
                == "Hate Speech"
            ).sum()
        )


        offensive_count = int(
            (
                history_df[
                    "Prediction"
                ]
                == "Offensive Language"
            ).sum()
        )


        safe_count = int(
            (
                history_df[
                    "Prediction"
                ]
                == "Neither"
            ).sum()
        )


        # ==========================================
        # METRICS
        # ==========================================

        metric1, metric2, metric3, metric4 = (
            st.columns(4)
        )


        metric1.metric(
            "Total Analyzed",
            total_analyzed
        )


        metric2.metric(
            "Hate Speech",
            hate_count
        )


        metric3.metric(
            "Offensive",
            offensive_count
        )


        metric4.metric(
            "Safe / Neither",
            safe_count
        )


        st.write("")


        # ==========================================
        # CLASS SUMMARY
        # ==========================================

        class_summary = (
            history_df[
                "Prediction"
            ]
            .value_counts()
            .reindex(
                [
                    "Hate Speech",
                    "Offensive Language",
                    "Neither"
                ],
                fill_value=0
            )
            .reset_index()
        )


        class_summary.columns = [
            "Class",
            "Count"
        ]


        dash_col1, dash_col2 = (
            st.columns(2)
        )


        with dash_col1:

            class_bar = px


# ==================================================
# MODERATION GUIDE
# ==================================================

st.divider()

with st.expander("🚦 View Moderation Guide"):

    st.write(
        "✅ **ALLOW** — Content appears safe with high confidence."
    )

    st.write(
        "⚠️ **WARN** — Offensive language has been detected."
    )

    st.write(
        "👤 **REVIEW** — The prediction is uncertain or potentially harmful "
        "and should be checked by a human moderator."
    )

    st.write(
        "🚫 **BLOCK** — High-confidence harmful or hate-speech content detected."
    )


# ==================================================
# MODEL INFORMATION
# ==================================================

with st.expander("🤖 About the AI Model"):

    st.write(
        "**Model:** DistilRoBERTa (`distilroberta-base`)"
    )

    st.write(
        "**Dataset:** Hate Speech and Offensive Language Dataset"
    )

    st.write("**Classification Classes:**")

    st.write("• Hate Speech")
    st.write("• Offensive Language")
    st.write("• Neither")

    st.write("**Test Accuracy:** 83.99%")
    st.write("**Macro Precision:** 0.6953")
    st.write("**Macro Recall:** 0.8176")
    st.write("**Macro F1-score:** 0.7196")
    st.write("**Training Device:** CPU")


# ==================================================
# DISCLAIMER
# ==================================================

st.divider()

st.caption(
    "⚠️ AI predictions and OCR extraction may occasionally be incorrect. "
    "Important or uncertain moderation decisions should include human review."
)







