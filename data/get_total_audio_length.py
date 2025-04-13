from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import wave
import torchaudio
import pympi
from typing import List, Tuple, Dict
import re
import glob
import tqdm


def get_segments_duration(eaf_file: str,
                          transcription_tier: str | list[str] = "default") -> int:
    eaf = pympi.Elan.Eaf(eaf_file)

    if isinstance(transcription_tier, str):
        transcription_tier = [transcription_tier]

    total_duration = 0
    for tier in transcription_tier:
        annotations = eaf.get_annotation_data_for_tier(tier)
        for start, end, annotation in annotations:
            total_duration += end - start
    
    return total_duration / 1000


def get_vad_duration(wav_file: str,
                     device: str = "cpu") -> int:
    # Load Silero VAD model
    # device = torch.device(device)  # Use CPU
    model = load_silero_vad()
    wav = read_audio(wav_file)
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        return_seconds=True,  # Return speech timestamps in seconds (default is samples)
        )

    # Compute total speech duration
    total_speech_duration = sum((ts['end'] - ts['start'])
                                for ts in speech_timestamps)
    # print(f"Total spoken duration: {total_speech_duration:.2f} seconds")

    return total_speech_duration


def get_wav_duration_torchaudio(file_path):
    wav, sample_rate = torchaudio.load(file_path)
    duration = wav.shape[1] / sample_rate
    return duration


def get_wav_duration_wave(file_path):
    with wave.open(file_path, "r") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        duration = frames / float(rate)
    return duration


def get_num_chars(annotations: List[Tuple[int, int, str]]) -> int:
    """Count the total number of characters in an eaf annotation.
    Ignore whitespaces.
    """
    transcript = "".join(annotation for _, _, annotation in annotations)
    transcript = re.sub("<ja>", "", transcript)
    transcript = re.sub("</ja>", "", transcript)
    transcript = re.sub("<dis>", "", transcript)
    transcript = re.sub("</dis>", "", transcript)
    transcript = re.sub("<unsure>", "", transcript)
    transcript = re.sub("</unsure>", "", transcript)
    return len(transcript)


def get_stats(wav_file: str,
              eaf_file: str,
              transcription_tier: str | list[str] = "default") -> None:
    """Get stats of wav and eaf."""
    # Audio
    total_dur = get_wav_duration_wave(wav_file)
    print(f"Total duration: {total_dur:.3f}")

    seg_dur = get_segments_duration(eaf_file,
                                    transcription_tier)
    print(f"Segment duration: {seg_dur:.3f} seconds")

    vad_dur = get_vad_duration(wav_file)
    print(f"VAD duration: {vad_dur:.3f} seconds")

    # Annotation
    eaf = pympi.Elan.Eaf(eaf_file)

    if isinstance(transcription_tier, str):
        transcription_tier = [transcription_tier]

    for tier in transcription_tier:
        print(f"Tier: {tier}")
        num_samples = len(eaf.get_annotation_data_for_tier(tier))
        print(f"Number of samples: {num_samples}")

        num_chars = get_num_chars(eaf.get_annotation_data_for_tier(tier))
        print(f"Number of characters: {num_chars}")
        

def get_audio_lengths(audio_folder: str = "./dictionary/ikmvoc") -> Dict[str, float]:
    """Get total raw audio duration and VAD duration."""
    files = glob.glob(audio_folder + "/*.wav")
    
    raw_length = 0
    vad_length = 0
    for wav_file in tqdm.tqdm(files):
        raw_length += get_wav_duration_wave(wav_file)
        vad_length += get_vad_duration(wav_file)
        
    return {"raw_length": raw_length,
            "vad_length": vad_length}
    

def main() -> None:
    stats = get_audio_lengths()
    print(f"Raw length: {stats['raw_length']:.3f}")
    print(f"VAD length: {stats['vad_length']:.3f}")
        

if __name__ == "__main__":
    main()