\# 🛡️ AI Hate Speech Moderation



An AI-powered social media content moderation system that detects \*\*Hate Speech\*\*, \*\*Offensive Language\*\*, and \*\*Safe/Neutral Content\*\* using a fine-tuned \*\*DistilRoBERTa\*\* model.



The application supports individual text analysis, batch moderation, image/screenshot OCR, Explainable AI, visual analytics, and automated moderation recommendations.



\---



\## 🌐 Live Application



🔗 \*\*Streamlit App:\*\*  

https://hatespeechmoderationai-usepds5payifbymfdrftta.streamlit.app/



🔗 \*\*GitHub Repository:\*\*  

https://github.com/prema-2811/HateSpeechModerationAI



\---



\## 🎯 Project Objective



Social media platforms receive huge volumes of posts, comments, messages, and user-generated content every day.



Manually reviewing all this content is difficult and time-consuming.



This project provides an AI-assisted moderation system that can automatically:



\- Detect harmful or offensive text

\- Estimate prediction confidence

\- Recommend moderation actions

\- Analyze multiple comments at once

\- Extract and moderate text from screenshots

\- Explain influential words behind predictions

\- Display moderation statistics visually



\---



\## ✨ Main Features



\### 🔎 1. Single Text Analysis



Users can enter a:



\- Social media post

\- Comment

\- Tweet

\- Message

\- Other textual content



The system provides:



\- Predicted class

\- Confidence score

\- Moderation action

\- Explanation

\- Confidence visualization

\- Explainable AI word analysis



\---



\### 📁 2. Batch File Analysis



Supports:



\- `.csv`

\- `.txt`



Multiple comments can be analyzed automatically.



For every entry, the system generates:



\- Prediction

\- Confidence

\- Moderation action

\- Explanation



Results can also be downloaded as a CSV moderation report.



\---



\### 🖼️ 3. Image / Screenshot Moderation



Supports:



\- PNG

\- JPG

\- JPEG

\- WEBP



The system uses \*\*Tesseract OCR\*\* to extract text from social media screenshots.



For screenshots containing multiple comments, the application attempts to separate the comments and analyze each comment individually.



Example flow:



```text

Instagram Screenshot

&#x20;       ↓

Tesseract OCR

&#x20;       ↓

Comment Extraction

&#x20;       ↓

DistilRoBERTa

&#x20;       ↓

Individual Predictions

&#x20;       ↓

Moderation Report

