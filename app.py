import streamlit as st

from src.predict import predict_text


# Page configuration
st.set_page_config(
    page_title="AI Hate Speech Moderation",
    page_icon="🛡️",
    layout="centered"
)


# Title
st.title("🛡️ AI Hate Speech Detection")
st.subheader("Social Media Content Moderation using RoBERTa")

st.write(
    """
    Enter a social media post or comment below.

    The RoBERTa model classifies the content as:

    **Hate Speech • Offensive Language • Neither**
    """
)


# User input
text = st.text_area(
    "Social Media Content",
    placeholder="Type a social media post or comment here...",
    height=150
)


# Analyze button
if st.button("Analyze Content", type="primary"):

    if not text.strip():

        st.warning("Please enter some text first.")

    else:

        with st.spinner("Analyzing content..."):

            result = predict_text(text)

        label = result["label"]
        confidence = result["confidence"] * 100
        action = result["action"]
        message = result["message"]

        st.divider()

        st.subheader("Analysis Result")

        st.write("### Prediction")
        st.write(label)

        st.write("### Confidence")
        st.write(f"{confidence:.2f}%")

        st.progress(
            min(
                int(confidence),
                100
            )
        )

        st.write("### Moderation Action")

        if action == "BLOCK":
            st.error("🚫 BLOCK")

        elif action == "WARN":
            st.warning("⚠️ WARN")

        elif action == "REVIEW":
            st.info("👤 REVIEW")

        else:
            st.success("✅ ALLOW")

        st.write("### Moderation Message")
        st.write(message)


st.divider()

st.subheader("Moderation Policy")

st.markdown(
    """
    - **BLOCK** — High-confidence hate speech
    - **WARN** — High-confidence offensive language
    - **REVIEW** — Uncertain or potentially harmful content
    - **ALLOW** — Content considered safe with high confidence
    """
)

st.caption(
    "AI predictions can be incorrect. Important moderation decisions "
    "should include human review."
)