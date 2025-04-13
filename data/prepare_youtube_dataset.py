from datasets import Audio, Dataset
import os
import glob
import pandas as pd
import argparse
import pympi
from typing import List, Dict, Tuple
from pydub import AudioSegment
import re
from tqdm import tqdm
import json

# import
from utils import hiragana_to_romaji, hiragana_to_phoneme


def get_wav_eaf_pairs(audio_dir: str) -> List[Dict[str, str]]:
    """Get pairs of wav and eaf files."""
    sub_dirs = glob.glob(os.path.join(audio_dir, "*"))

    pairs = []
    for d in sub_dirs:
        if os.path.basename(d) == "segments":
            continue
        
        wav_files = glob.glob(os.path.join(d, "*.wav"))
        eaf_files = glob.glob(os.path.join(d, "*.eaf"))

        if len(wav_files) != 1 or len(eaf_files) != 1:
            print(f"Skipping directory {d} due to unexpected number of files.")
            continue

        wav_file = wav_files[0]
        eaf_file = eaf_files[0]

        pairs.append({"wav": wav_file,
                      "eaf": eaf_file})

    return pairs


def get_transcription(eaf_file: str | List[str],
                      transcription_tier: str = "default") -> List[Dict[str, str | int]]:
    """Get the transcription from an eaf file."""
    eaf = pympi.Elan.Eaf(eaf_file)
    
    if isinstance(transcription_tier, str):
        transcription_tier = [transcription_tier]
    
    for tier in transcription_tier:
        annotations = eaf.get_annotation_data_for_tier(tier)
        annotations = [
            {"start": start,
             "end": end,
             "transcription": transcription}
            for start, end, transcription in annotations
        ]
    
    return annotations


TAGS = ["ja", "dis", "unsure"]


def remove_tags(text: str,
                tags: List[str]) -> str:
    """Remove XML tags from text."""
    for tag in tags:
        text = text.replace(f"<{tag}>", "").replace(f"</{tag}>", "")
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)
        
    return text.strip()


def make_audio_data_from_splits(audio: AudioSegment,
                                annotations: List[Dict[str, str | int]],
                                segments_subdir: str,
                                audio_name: str,
                                romaji_mapping: dict,
                                phoneme_mapping: dict) -> dict:
    """Make a dictionary of audio data from the splits."""
    audios = []
    transcriptions = []
    romajis = []
    phonemes = []
    starts = []
    ends = []
    titles = []
    
    for annotation in tqdm(annotations):
        start = annotation["start"]
        end = annotation["end"]
        transcription = annotation["transcription"]
        
        # Get the audio segment
        segment = audio[start:end]
        
        # Save the audio segment
        segment_name = f"{audio_name}_{start}_{end}.wav"
        segment_path = os.path.join(segments_subdir, segment_name)
        segment.export(segment_path, format="wav")
        
        # Clean the transcription
        transcription = remove_tags(transcription, TAGS)
        romaji = hiragana_to_romaji(transcription, romaji_mapping)
        phoneme = hiragana_to_phoneme(transcription, phoneme_mapping)
        
        # Add the segment name, transcription, start, end, and audio to the data
        audios.append(segment_path)
        transcriptions.append(transcription)
        romajis.append(romaji)
        phonemes.append(phoneme)
        starts.append(start)
        ends.append(end)
        titles.append(audio_name)
        
    audio_data = {
        "audio": audios,
        "transcription": transcriptions,
        "romaji": romajis,
        "phoneme": phonemes,
        "start": starts,
        "end": ends,
        "title": titles
    }
    
    return audio_data


def convert_to_romaji(text: str,
                      mapping: dict) -> str:
    """Convert kana to romaji."""
    
    

def get_args() -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare a dictionary dataset."
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="./youtube/",
        help=(
            "Path to the directory containing audio files. "
            "This should contain the sound files referenced in the dictionary."
        )
    )
    parser.add_argument(
        "--romaji_map_file",
        type=str,
        default="../src/romaji_map.json",
        help="Path to the romaji mapping file."
    )
    parser.add_argument(
        "--phoneme_map_file",
        type=str,
        default="../src/phoneme_map.json",
        help="Path to the phoneme mapping file."
    )
    parser.add_argument(
        "--push_to_hub",
        action='store_true',
        help=(
            "Whether to upload the dataset to Hugging Face Hub. "
            "If set, the dataset will be uploaded after preparation."
            "Make sure to set the `repo_name` in the script accordingly."
        )
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default="ikema_youtube_asr",
        help=(
            "The name of the repository to use for uploading the dataset to Hugging Face. "
            "This is only used if `--upload_to_hf` is set. "
            "Make sure to create the repo on Hugging Face before uploading."
        )
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    with open(args.romaji_map_file, "r") as f:
        romaji_mapping = json.load(f)
    with open(args.phoneme_map_file, "r") as f:
        phoneme_mapping = json.load(f)
    
    wav_eaf_pairs = get_wav_eaf_pairs(args.audio_dir)
    if not wav_eaf_pairs:
        print("No valid wav-eaf pairs found. Exiting.")
        exit(1)
    
    # Make a new dir for the segments    
    segments_dir = os.path.join(args.audio_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    
    dataset_dict = {
        "audio": [],
        "transcription": [],
        "romaji": [],
        "phoneme": [],
        "start": [],
        "end": [],
        "title": []
    }
        
    for pair in wav_eaf_pairs:
        wav_file = pair["wav"]
        eaf_file = pair["eaf"]
        
        audio = AudioSegment.from_wav(wav_file)
        audio_name = os.path.basename(wav_file)
        audio_name = os.path.splitext(audio_name)[0]
        print(f"Processing {audio_name}...")
        
        segments_subdir = os.path.join(segments_dir, audio_name)
        os.makedirs(segments_subdir, exist_ok=True)
        
        # Get the transcription
        if audio_name == "kaichou_shokureki":
            mono_audios = audio.split_to_mono()
            
            # create subdir for tadashi and hiroyuki
            print("Processing tadashi...")
            tadashi_subdir = f"{args.audio_dir}/{audio_name}/tadashi"
            os.makedirs(tadashi_subdir, exist_ok=True)
            
            # tadashi
            tadashi_audio_file = mono_audios[0].export(
                f"{tadashi_subdir}/{audio_name}_tadashi.wav",
                format="wav"
            )
            tadashi_audio = AudioSegment.from_wav(tadashi_audio_file)
            annotations_tadashi = get_transcription(
                eaf_file,
                transcription_tier="tadashi"
            )
            data_tadashi = make_audio_data_from_splits(
                audio=tadashi_audio,
                annotations=annotations_tadashi,
                segments_subdir=segments_subdir,
                audio_name=f"{audio_name}_tadashi",
                romaji_mapping=romaji_mapping,
                phoneme_mapping=phoneme_mapping
            )
            
            # Tests
            assert len(data_tadashi["audio"]) == len(data_tadashi["transcription"]), \
                f"Audio and transcription lengths do not match for {audio_name}_tadashi."
            assert len(data_tadashi["audio"]) == len(data_tadashi["start"]), \
                f"Audio and start times lengths do not match for {audio_name}_tadashi."
            assert len(data_tadashi["audio"]) == len(data_tadashi["end"]), \
                f"Audio and end times lengths do not match for {audio_name}_tadashi."
            assert len(data_tadashi["audio"]) == len(data_tadashi["title"]), \
                f"Audio and titles lengths do not match for {audio_name}_tadashi."
                
            # Combine the data
            dataset_dict["audio"].extend(data_tadashi["audio"])
            dataset_dict["transcription"].extend(data_tadashi["transcription"])
            dataset_dict["romaji"].extend(data_tadashi["romaji"])
            dataset_dict["phoneme"].extend(data_tadashi["phoneme"])
            dataset_dict["start"].extend(data_tadashi["start"])
            dataset_dict["end"].extend(data_tadashi["end"])
            dataset_dict["title"].extend(data_tadashi["title"])
            
            # hiroyuki
            print("Processing hiroyuki...")
            hiroyuki_subdir = f"{args.audio_dir}/{audio_name}/hiroyuki"
            os.makedirs(hiroyuki_subdir, exist_ok=True)
            
            hiroyuki_audio_file = mono_audios[1].export(
                f"{hiroyuki_subdir}/{audio_name}_hiroyuki.wav",
                format="wav"
            )
            hiroyuki_audio = AudioSegment.from_wav(hiroyuki_audio_file)
            annotations_hiroyuki = get_transcription(
                eaf_file,
                transcription_tier="hiroyuki"
            )
            data_hiroyuki = make_audio_data_from_splits(
                audio=hiroyuki_audio,
                annotations=annotations_hiroyuki,
                segments_subdir=segments_subdir,
                audio_name=f"{audio_name}_hiroyuki",
                romaji_mapping=romaji_mapping,
                phoneme_mapping=phoneme_mapping
            )
            
            # Tests
            assert len(data_hiroyuki["audio"]) == len(data_hiroyuki["transcription"]), \
                f"Audio and transcription lengths do not match for {audio_name}_hiroyuki."
            assert len(data_hiroyuki["audio"]) == len(data_hiroyuki["start"]), \
                f"Audio and start times lengths do not match for {audio_name}_hiroyuki."
            assert len(data_hiroyuki["audio"]) == len(data_hiroyuki["end"]), \
                f"Audio and end times lengths do not match for {audio_name}_hiroyuki."
            assert len(data_hiroyuki["audio"]) == len(data_hiroyuki["title"]), \
                f"Audio and titles lengths do not match for {audio_name}_hiroyuki."
                
            # Combine the data
            dataset_dict["audio"].extend(data_hiroyuki["audio"])
            dataset_dict["transcription"].extend(data_hiroyuki["transcription"])
            dataset_dict["romaji"].extend(data_hiroyuki["romaji"])
            dataset_dict["phoneme"].extend(data_hiroyuki["phoneme"])
            dataset_dict["start"].extend(data_hiroyuki["start"])
            dataset_dict["end"].extend(data_hiroyuki["end"])
            dataset_dict["title"].extend(data_hiroyuki["title"])
            
        else:
            # Get the transcription
            transcription_tier = "default"
            annotations = get_transcription(eaf_file,
                                            transcription_tier=transcription_tier)
            if not annotations:
                print(f"No annotations found in {eaf_file}. Skipping.")
                continue
            
            data = make_audio_data_from_splits(
                audio=audio,
                annotations=annotations,
                segments_subdir=segments_subdir,
                audio_name=audio_name,
                romaji_mapping=romaji_mapping,
                phoneme_mapping=phoneme_mapping
            )

            # Tests
            assert len(data["audio"]) == len(data["transcription"]), \
                f"Audio and transcription lengths do not match for {audio_name}."
            assert len(data["audio"]) == len(data["start"]), \
                f"Audio and start times lengths do not match for {audio_name}."
            assert len(data["audio"]) == len(data["end"]), \
                f"Audio and end times lengths do not match for {audio_name}."
            assert len(data["audio"]) == len(data["title"]), \
                f"Audio and titles lengths do not match for {audio_name}."
                
            # Combine the data
            dataset_dict["audio"].extend(data["audio"])
            dataset_dict["transcription"].extend(data["transcription"])
            dataset_dict["romaji"].extend(data["romaji"])
            dataset_dict["phoneme"].extend(data["phoneme"])
            dataset_dict["start"].extend(data["start"])
            dataset_dict["end"].extend(data["end"])
            dataset_dict["title"].extend(data["title"])
            
            print(f"Processed {audio_name} with {len(data['audio'])} segments.")
        
    # Create an audio dataset
    audio_dataset = Dataset.from_dict(dataset_dict).cast_column("audio", Audio(sampling_rate=16000))
    
    if args.push_to_hub:
        # Upload the dataset to Hugging Face Hub
        audio_dataset.push_to_hub(args.repo_name)
        print(f"Dataset uploaded to Hugging Face Hub at: {args.repo_name}")
    else:
        # Save locally
        audio_dataset.save_to_disk(args.repo_name)
        print(f"Dataset saved locally at: {args.repo_name}")