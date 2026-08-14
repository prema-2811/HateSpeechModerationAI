import streamlit as st
from src.predict import predict_text


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="AI Hate Speech Moderation",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# SUBTLE BACKGROUND
# --------------------------------------------------
# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🛡️ AI Hate Speech Moderation")

st.markdown(
    "### Hate Speech & Offensive Language Detection using RoBERTa"
)

st.write(
    "Analyze social-media posts, comments, tweets, or messages "
    "and receive an AI-assisted moderation recommendation."
)

st.divider()
st.markdown(
    """
    <style>

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                circle at top left,
                #dbeafe 0%,
                transparent 35%
            ),
            radial-gradient(
                circle at bottom right,
                #e0e7ff 0%,
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #f8fafc 0%,
                #eef2ff 50%,
                #f0f9ff 100%
            );
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)# --------------------------------------------------
# INPUT
# --------------------------------------------------

st.subheader("📝 Analyze Social Media Content")

st.caption(
    "Enter or paste the content you want to analyze."
)

text = st.text_area(
    "Social Media Content",
    placeholder="Type a post, tweet, comment or message here...",
    height=150,
    label_visibility="collapsed"
)

analyze_button = st.button(
    "🔍 Analyze Content",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

if analyze_button:

    if not text.strip():

        st.warning("⚠️ Please enter some content first.")

    else:

        with st.spinner("RoBERTa is analyzing the content..."):

            result = predict_text(text)

        label = result["label"]
        confidence = result["confidence"] * 100
        action = result["action"]
        message = result["message"]

        st.divider()

        st.header("📊 Analysis Result")

        col1, col2, col3 = st.columns(3)


        # ------------------------------------------
        # PREDICTION
        # ------------------------------------------

        with col1:

            with st.container(border=True):

                st.caption("🔎 PREDICTION")

                if label == "Hate Speech":
                    st.error(f"### {label}")

                elif label == "Offensive Language":
                    st.warning(f"### {label}")

                else:
                    st.success(f"### {label}")


        # ------------------------------------------
        # CONFIDENCE
        # ------------------------------------------

        with col2:

            with st.container(border=True):

                st.caption("📊 MODEL CONFIDENCE")

                st.metric(
                    label="Confidence",
                    value=f"{confidence:.2f}%",
                    label_visibility="collapsed"
                )

                st.progress(
                    min(int(confidence), 100)
                )


        # ------------------------------------------
        # MODERATION ACTION
        # ------------------------------------------

        with col3:

            with st.container(border=True):

                st.caption("🛡️ MODERATION ACTION")

                if action == "BLOCK":

                    st.error("### 🚫 BLOCK")

                elif action == "WARN":

                    st.warning("### ⚠️ WARN")

                elif action == "REVIEW":

                    st.info("### 👤 REVIEW")

                else:

                    st.success("### ✅ ALLOW")


        # ------------------------------------------
        # EXPLANATION
        # ------------------------------------------

        st.write("")

        with st.container(border=True):

            st.subheader("💡 Why this decision?")

            st.write(message)


# --------------------------------------------------
# MODERATION GUIDE - LESS PROMINENT
# --------------------------------------------------

st.divider()

with st.expander("🚦 View Moderation Guide"):

    st.caption(
        "Meaning of the moderation actions used by the system."
    )

    guide1, guide2, guide3, guide4 = st.columns(4)

    with guide1:

        st.success("✅ **ALLOW**")

        st.caption(
            "Safe content with high confidence."
        )


    with guide2:

        st.warning("⚠️ **WARN**")

        st.caption(
            "Offensive content detected."
        )


    with guide3:

        st.info("👤 **REVIEW**")

        st.caption(
            "Requires human moderation."
        )


    with guide4:

        st.error("🚫 **BLOCK**")

        st.caption(
            "High-risk harmful content."
        )


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

with st.expander("🤖 About the AI Model"):

    st.write("**Model:** DistilRoBERTa")

    st.write(
        "**Classes:** Hate Speech, Offensive Language, Neither"
    )

    st.write("**Test Accuracy:** 83.99%")

    st.write("**Macro F1-score:** 0.7196")

    st.write("**Training:** CPU-based fine-tuning")


# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------

st.divider()

st.caption(
    "⚠️ AI predictions may be incorrect. "
    "Important or uncertain moderation decisions "
    "should include human review."
)