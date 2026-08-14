\# AI Agent for Hate Speech Detection and Content Moderation in Social Media



An AI-powered content moderation system that detects harmful social-media content using a fine-tuned \*\*RoBERTa-family transformer model\*\* and recommends moderation actions.



\## Problem Statement



Social-media platforms contain large amounts of user-generated content, including hate speech and offensive language. Manual moderation is difficult and time-consuming.



This project uses \*\*RoBERTa\*\* to automatically classify social-media text and assist content moderation.



The system classifies text into:



\* Hate Speech

\* Offensive Language

\* Neither



It then recommends one of four moderation actions:



\* BLOCK

\* WARN

\* REVIEW

\* ALLOW



\## Dataset



Dataset: \*\*Hate Speech and Offensive Language Dataset\*\*



Source:



https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset



Total records:



\*\*24,783\*\*



Classes:



\* `0` - Hate Speech

\* `1` - Offensive Language

\* `2` - Neither



The dataset is highly imbalanced, so \*\*Macro F1-score\*\* is used as an important evaluation metric.



\## Data Split



The dataset was divided using stratified sampling:



\* Training: 19,826 samples

\* Validation: 2,478 samples

\* Test: 2,479 samples



\## Model



The final model used in this project is:



\*\*DistilRoBERTa (`distilroberta-base`)\*\*



It is a lightweight RoBERTa-family transformer suitable for CPU-based fine-tuning and inference.



Training configuration:



\* Maximum sequence length: 64

\* Batch size: 16

\* Epochs: 2

\* Learning rate: 2e-5

\* Device: CPU

\* Loss function: Weighted Cross Entropy



Class weights are used because the dataset contains significantly fewer hate-speech examples than offensive-language examples.



\## Model Performance



Performance on the unseen test dataset:



| Metric          |  Score |

| --------------- | -----: |

| Accuracy        | 83.99% |

| Macro Precision | 0.6953 |

| Macro Recall    | 0.8176 |

| Macro F1-score  | 0.7196 |



\### Class-wise Performance



| Class              | Precision | Recall | F1-score |

| ------------------ | --------: | -----: | -------: |

| Hate Speech        |      0.25 |   0.70 |     0.37 |

| Offensive Language |      0.98 |   0.83 |     0.90 |

| Neither            |      0.86 |   0.92 |     0.89 |



\## Content Moderation Agent



The classifier is combined with a rule-based moderation layer.



The model first predicts the content category and confidence score.



The moderation agent then recommends:



\### BLOCK



High-confidence hate speech.



\### WARN



High-confidence offensive language.



\### REVIEW



Potentially harmful or uncertain content that should be checked by a human moderator.



\### ALLOW



Content predicted as safe with high confidence.



The REVIEW option helps reduce the risk of automatically allowing or blocking uncertain predictions.



\## Project Structure



```text

HateSpeechModerationAI/

│

├── data/

│   ├── raw/

│   └── processed/

│

├── models/

│   └── roberta/

│

├── results/

│   └── metrics.json

│

├── src/

│   ├── \_\_init\_\_.py

│   ├── preprocess.py

│   ├── train.py

│   ├── evaluate.py

│   ├── predict.py

│   └── moderation\_agent.py

│

├── app.py

├── config.py

├── requirements.txt

├── .gitignore

└── README.md

```



\## Technologies Used



\* Python

\* PyTorch

\* Hugging Face Transformers

\* RoBERTa

\* Pandas

\* NumPy

\* Scikit-learn

\* Gradio



\## Installation



Create a virtual environment:



```bash

python -m venv venv

```



Activate it on Windows:



```bash

venv\\Scripts\\activate

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\## Dataset Preprocessing



Place `labeled\_data.csv` inside:



```text

data/raw/

```



Then run:



```bash

python src/preprocess.py

```



This creates the training, validation, and test datasets.



\## Training



Train the RoBERTa model using:



```bash

python src/train.py

```



The best model is saved inside:



```text

models/roberta/

```



\## Evaluation



Evaluate the trained model using:



```bash

python src/evaluate.py

```



\## Command-Line Prediction



Run:



```bash

python src/predict.py

```



Enter any social-media post or comment to receive:



\* Prediction

\* Confidence score

\* Moderation action

\* Moderation message



\## Run the Web Application



Start the Gradio application:



```bash

python app.py

```



Then open:



```text

http://127.0.0.1:7860

```



\## Limitations



The dataset is highly imbalanced, particularly for the Hate Speech class.



Because of this, the model may sometimes confuse offensive language with hate speech or incorrectly classify potentially harmful content as safe.



For this reason, uncertain cases are sent to \*\*human review\*\* instead of relying entirely on automatic moderation.



\## Future Improvements



Possible future improvements include:



\* Training with a larger and more balanced hate-speech dataset

\* Multilingual hate-speech detection

\* Context-aware moderation

\* Continuous feedback from human moderators

\* Improved handling of sarcasm and implicit hate speech



\## Disclaimer



This system is intended as an educational AI content-moderation project. Automated moderation predictions may be incorrect and should not replace human judgment for important moderation decisions.



