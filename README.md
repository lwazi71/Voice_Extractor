# Voice_Extractor


Extract high-quality voice segments of a specific target speaker from complex audio environments. This tool employs a sophisticated pipeline leveraging state-of-the-art machine learning models for speaker diarization, vocal separation, speaker verification, and transcription.


pip install -r requirements.txt



python run_extractor.py `
    --input-audio "C:\Users\maste\Desktop\FT_audio\voice_extractor\input\full_audio.wav" `
    --reference-audio "C:\Users\maste\Desktop\FT_audio\voice_extractor\input\target_sample.wav" `
    --target-name "Scarlett" `
    --output-base-dir "C:\Users\maste\Desktop\FT_audio\voice_extractor\output" `
    --token "hf_changeME" `
    --osd-model "pyannote/overlapped-speech-detection" `
    --dry-run `
    --debug
