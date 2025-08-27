import assemblyai as aai
import pandas as pd

# Set your AssemblyAI API key
aai.settings.api_key = ""

def transcribe_and_diarize(audio_path: str) -> pd.DataFrame:
    transcriber = aai.Transcriber()

    transcript = transcriber.transcribe(
        audio_path,
        config=aai.TranscriptionConfig(speaker_labels=True)
    )

    rows = []
    for utt in transcript.utterances:
        speaker = utt.speaker  # e.g., "SPEAKER_00", "SPEAKER_01"
        start = f"{utt.start / 1000:.2f}s"
        end = f"{utt.end / 1000:.2f}s"
        rows.append({
            "Speaker": speaker,
            "Timestamp": f"{start} - {end}",
            "Text": utt.text
        })

    return pd.DataFrame(rows)
