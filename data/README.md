## Files
- `get_total_audio_length.py`: This program measures the total audio duration (length) of the dictionary dataset. In addition to the raw audio length, it also measures the duration of the true spoken segments using VAD (voice activity detection).

## Total audio length of the dictionary dataset
Raw length: 5017.932
VAD length: 3328.900

## Notes
- In order to run `get_total_audio_length.py`, the function `get_vad_duration()` relies on `silero-vad`, which uses `torchaudio`.
The current `torchaudio` version does not support `ffmpeg>=7`, so you might need to downgrade your version.
If you are using MacOS and do not really need `ffmpeg=7`, then here's a quick fix:
```
# Uninstall ffmpeg
brew uninstall --force ffmpeg
brew cleanup -s ffmpeg
brew cleanup --prune-prefix

# Reinstall it
brew install ffmpeg@6

# Link it into your system's default path
brew link --force --overwrite ffmpeg@6
```