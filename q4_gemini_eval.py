from google import genai
from google.genai import types
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
)

# API configuration taken from: https://ai.google.dev/gemini-api/docs/quickstart?hl=es-419
# Reasoning logic taken from: https://ai.google.dev/gemini-api/docs/thinking?hl=es-419
        
GEMINI_API_KEY = "AIzaSyANge03Cu83ioyMtavor3UqBcAiC2AgH3s" 

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "models/gemma-3-4b-it"

def evaluate_model(y_true, y_pred, model_name, output_path="reports/"):
   
    os.makedirs(output_path, exist_ok=True)
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
    }

    report = classification_report(y_true, y_pred, output_dict=True)
    avg_metrics = {
        "Precision (macro)": report["macro avg"]["precision"],
        "Recall (macro)": report["macro avg"]["recall"],
        "F1 (macro)": report["macro avg"]["f1-score"],
        "Precision (weighted)": report["weighted avg"]["precision"],
        "Recall (weighted)": report["weighted avg"]["recall"],
        "F1 (weighted)": report["weighted avg"]["f1-score"]
    }

    all_stats = {**metrics, **avg_metrics}

    
    fig, axes = plt.subplots(2, 1, figsize=(10, 16)) 
    fig.suptitle(f'Model Performance Report: {model_name}', fontsize=18, fontweight='bold')

    sns.barplot(x=list(all_stats.values()), y=list(all_stats.keys()), palette='magma', ax=axes[0])
    axes[0].set_xlim(0, 1.1)
    axes[0].set_title('Metric Overview (Higher is Better)', fontsize=14)

    for i, v in enumerate(all_stats.values()):
        axes[0].text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], annot_kws={"size": 14},
                xticklabels=["Neg", "Neu", "Pos"], 
                yticklabels=["Neg", "Neu", "Pos"])
    axes[1].set_title('Confusion Matrix', fontsize=14)
    axes[1].set_ylabel('Actual Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    save_path = f"{output_path}/{model_name.replace(' ', '_')}_Adapted_Report.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


try:
    df_test = pd.read_csv("test_df_processed.csv")
    df_test = df_test.head(1000) 
except:
    print("Error loading test dataset.")
    exit()

def get_gemini_pred(text):
    prompt = f"""
    Act as a sentiment analysis expert. Your task is to classify the sentiment of a tweet.
    
    Here are clear examples of how you must classify (learn from these):

    Tweet: "So many farewell parties! sad to see people leaving .."
    Sentiment: negative

    Tweet: "happy birthday ness!!"
    Sentiment: positive

    Tweet: "I`ve got cups but you gotta come get them"
    Sentiment: neutral

    Tweet: "- Still a pity it comes with no lamb though"
    Sentiment: negative

    Tweet: "TAKE THAT, TAKE THAT!!!! IN YOUR FACES!!!!!!!!!!!!  Robbie won!"
    Sentiment: positive

    Tweet: "charlie and the chocolate factory, in the mood for some johnny depp, then bed."
    Sentiment: neutral

    ---
    Now, classify the following tweet. 
    Tweet: "{text}"
    Sentiment (respond with one word only: positive, negative, or neutral):
    """
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        prediccion = response.text.strip().lower()        
        prediccion = prediccion.replace(".", "").replace("\n", "")
        
        return prediccion
    except Exception as e:
        return "neutral"


y_pred_text = []

for i, row in df_test.iterrows():
    pred = get_gemini_pred(row['text'])
    y_pred_text.append(pred)
    time.sleep(3) 
    if i % 5 == 0:
        print(f"Processing {i}...")

label_map = {"negative": 0, "neutral": 1, "positive": 2}

y_pred = [label_map.get(x, 1) for x in y_pred_text]

if df_test['sentiment'].dtype == 'O':
    y_true = df_test['sentiment'].map(label_map).fillna(1).astype(int)
else:
    y_true = df_test['sentiment'].astype(int)

evaluate_model(
    y_true=y_true,
    y_pred=y_pred,
    model_name="Gemini gemma-3-4b-it",
)

