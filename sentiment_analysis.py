from transformers import pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load once
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=" cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True
)

def add_sentiment_column(df: pd.DataFrame) -> pd.DataFrame:
    def get_sentiment(text):
        result = sentiment_pipeline(text[:512])[0]
        return result["label"].upper()  # Ensures it's UPPER CASE
    df["Sentiment"] = df["Text"].apply(get_sentiment)
    return df


def plot_overall_sentiment_by_speaker(df: pd.DataFrame, output_path: str):
    sentiment_counts = df[df["Sentiment"] != "NEUTRAL"]  # ❌ remove NEUTRAL
    sentiment_counts = sentiment_counts.groupby(["Speaker", "Sentiment"]).size().reset_index(name="Count")

    if not sentiment_counts.empty:
        plt.figure(figsize=(8, 6))
        sns.barplot(data=sentiment_counts, x="Speaker", y="Count", hue="Sentiment")
        plt.title("Overall Sentiment Distribution by Speaker (No Neutral)")
        plt.ylabel("Number of Turns")
        plt.xlabel("Speaker")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "overall_sentiment_per_speaker.png"))
        plt.close()
    else:
        print("⚠️ Skipping overall sentiment plot — all values were NEUTRAL.")

def plot_sentiment_timeline(df, output_path):
    """
    Plots sentiment fluctuation with separate trend lines for each speaker.
    """
    sentiment_score_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
    df["SentimentScore"] = df["Sentiment"].map(sentiment_score_map)
    df["Exchange"] = range(1, len(df) + 1)

    plt.figure(figsize=(12, 6))

    # Colors for each speaker
    speaker_colors = {
        speaker: color for speaker, color in zip(df["Speaker"].unique(), ["blue", "orange", "green", "purple"])
    }

    for speaker in df["Speaker"].unique():
        speaker_df = df[df["Speaker"] == speaker]

        # Scatter plot for speaker
        plt.scatter(speaker_df["Exchange"], speaker_df["SentimentScore"],
                    color=speaker_colors[speaker], label=f"{speaker} Data Points", alpha=0.6)

        # Fit and plot trend line
        if len(speaker_df) >= 2:  # Need at least 2 points to fit a line
            z = np.polyfit(speaker_df["Exchange"], speaker_df["SentimentScore"], 3)
            p = np.poly1d(z)
            plt.plot(speaker_df["Exchange"], p(speaker_df["Exchange"]),
                     color=speaker_colors[speaker], linestyle="-", linewidth=2,
                     label=f"{speaker} Trend Line")

    # Neutral baseline
    plt.axhline(y=0, color="gray", linestyle="dashed", linewidth=1.5, label="Neutral Baseline")

    # Labels
    plt.xlabel("Conversation Exchange")
    plt.ylabel("Sentiment Score (-1: Negative, 0: Neutral, 1: Positive)")
    plt.title("Sentiment Fluctuation Over Conversation by Speaker")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save
    plt.savefig(os.path.join(output_path, "sentiment_trendline_per_speaker.png"))
    plt.close()
