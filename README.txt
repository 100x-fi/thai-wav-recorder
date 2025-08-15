Thai TTS Dataset Helper Tools
================================
1) thai_ljs_recorder.py
   - Gradio web app to record utterances one by one.
   - Saves WAV to ./wavs and appends to ./metadata.csv.
   - Usage:
       pip install gradio soundfile numpy scipy librosa
       python thai_ljs_recorder.py
     Optional: create prompts.txt (one sentence per line) and upload in the app to guide reading.

2) audacity_labels_to_ljs.py
   - If you prefer recording a long file and slicing in Audacity:
       - Create a label track with transcripts.
       - Export Multiple -> WAV files (utt0001.wav, ...)
       - Export Labels -> labels.txt
       - Run:
           python audacity_labels_to_ljs.py labels.txt
       - This writes metadata.csv compatible with F5-TTS.

Both scripts produce the LJSpeech-style layout expected by the finetune UI:
    metadata.csv   (utt_id|text)
    wavs/uttXXXX.wav
