import replicate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ------------------ CONFIG ------------------

INTENT_MODEL = "georgedavila/bart-large-mnli-classifier:d929487cf059f96a17752ebe55ae5a85b2e8be6cd627078e49c6caa2fd4213db"
SUMMARY_MODEL = "ibm-granite/granite-3.2-8b-instruct"

CANDIDATE_LABELS = [
    "Greeting", "Identification", "Request_Refill", "Express_Issue",
    "Clarification_Request", "Provide_Information", "Acknowledge",
    "Close_Conversation", "Check_Record", "Offer_Solution"
]

INTENT_TO_CATEGORY = {
    "Greeting": "Greeting Intent",
    "Identification": "Informational Intent",
    "Request_Refill": "Transactional Intent",
    "Express_Issue": "Complaint Intent",
    "Clarification_Request": "Support Intent",
    "Provide_Information": "Informational Intent",
    "Acknowledge": "Acknowledgment Intent",
    "Close_Conversation": "Closing Intent",
    "Check_Record": "Navigational Intent",
    "Offer_Solution": "Assurance Intent"
}

# ------------------ INTENT CLASSIFICATION ------------------

def classify_intents_with_replicate(text_list, candidate_labels):
    intent_results = []
    for text in text_list:
        try:
            response = replicate.run(
                INTENT_MODEL,
                input={
                    "text": text,
                    "labels": ", ".join(candidate_labels)
                }
            )
            top_intent = response[0]["label"] if response else "Unknown"
        except Exception as e:
            print("⚠️ Replicate intent API error:", e)
            top_intent = "Unknown"

        intent_results.append(top_intent)

    return intent_results


def annotate_intents_from_csv(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)
    print("🔍 Running intent classification via Replicate...")

    df["Intent"] = classify_intents_with_replicate(df["Text"].tolist(), CANDIDATE_LABELS)
    df["IntentCategory"] = df["Intent"].map(INTENT_TO_CATEGORY)

    df.to_csv(output_csv_path, index=False)
    print(f"✅ Annotated intent data saved to: {output_csv_path}")
    return df

# ------------------ VISUALIZATION ------------------

def plot_intent_distribution_by_speaker(df, output_dir):
    intent_counts = df.groupby(["IntentCategory", "Speaker"]).size().reset_index(name="Count")
    intent_counts = intent_counts[intent_counts["IntentCategory"].notna()]

    plt.figure(figsize=(10, 6))
    sns.barplot(data=intent_counts, x="IntentCategory", y="Count", hue="Speaker", palette="Set2")
    plt.title("Intent Category Distribution by Speaker")
    plt.ylabel("Count")
    plt.xlabel("Intent Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "intent_distribution_by_speaker.png")
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Saved intent distribution plot to: {output_path}")

# ------------------ SUMMARIZATION ------------------

def summarize_intents(df: pd.DataFrame) -> dict:
    customer_text = " ".join(df[df["Speaker"] == "SPEAKER_00"]["Text"].tolist())
    rep_text = " ".join(df[df["Speaker"] == "SPEAKER_01"]["Text"].tolist())

    try:
        issue_summary = " ".join(replicate.run(
            SUMMARY_MODEL,
            input={
                "prompt": f"Summarize the main issue the customer is facing:\n{customer_text}"
            }
        ))
        resolution_summary = " ".join(replicate.run(
            SUMMARY_MODEL,
            input={
                "prompt": f"Summarize how the representative helped resolve the issue:\n{rep_text}"
            }
        ))
        return {
            "issue_summary": issue_summary,
            "resolution_summary": resolution_summary
        }
    except Exception as e:
        print("⚠️ Replicate summarization API error:", e)
        return {
            "issue_summary": "",
            "resolution_summary": ""
        }
