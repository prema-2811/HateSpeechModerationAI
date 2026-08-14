import gradio as gr

from src.predict import predict_text


def analyze_content(text):

    if not text or not text.strip():
        return (
            "Please enter some text.",
            "-",
            "-",
            "Enter a social media post or comment first."
        )

    result = predict_text(text)

    label = result["label"]
    confidence = f"{result['confidence'] * 100:.2f}%"
    action = result["action"]
    message = result["message"]

    return label, confidence, action, message


with gr.Blocks(title="AI Hate Speech Moderation") as app:

    gr.Markdown(
        """
        # 🛡️ AI Hate Speech Detection & Content Moderation

        Enter a social-media post or comment below.

        The system uses a fine-tuned **RoBERTa model** to classify the content as:

        **Hate Speech · Offensive Language · Neither**

        It then recommends a moderation action.
        """
    )

    text_input = gr.Textbox(
        label="Social Media Content",
        placeholder="Type a comment or post here...",
        lines=5
    )

    analyze_button = gr.Button(
        "Analyze Content",
        variant="primary"
    )

    gr.Markdown("## Analysis Result")

    prediction_output = gr.Textbox(
        label="Prediction",
        interactive=False
    )

    confidence_output = gr.Textbox(
        label="Confidence",
        interactive=False
    )

    action_output = gr.Textbox(
        label="Moderation Action",
        interactive=False
    )

    message_output = gr.Textbox(
        label="Moderation Message",
        interactive=False
    )

    analyze_button.click(
        fn=analyze_content,
        inputs=text_input,
        outputs=[
            prediction_output,
            confidence_output,
            action_output,
            message_output
        ]
    )

    gr.Markdown(
        """
        ### Moderation Policy

        - **BLOCK** → High-confidence hate speech
        - **WARN** → High-confidence offensive language
        - **REVIEW** → Uncertain or potentially harmful content
        - **ALLOW** → No harmful content detected

        **Note:** AI predictions can be incorrect. Important moderation decisions should include human review.
        """
    )


if __name__ == "__main__":
    app.launch()