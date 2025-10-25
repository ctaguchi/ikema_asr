from datasets import Audio, Dataset, DatasetDict
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

# local import
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import hiragana_to_romaji, hiragana_to_phoneme, remove_tags


TEST_AUDIO = {
    "jugon",
    "mimamuibusu"
}


title_map = {
    # I0045_若いころの話 is unannotated, but this is identical to tamunu
    # I0045_小エビ取り is unannotated
    "yaafutsu": "I0045_yaafucI", # this is not yet included in the dataset
    # I0045_keito_namako is unannotated
    "utakinohanashi": "I0096_nimaigai_utaki", # this is not yet included in the dataset
    "tamunu": "I0045_tamunu", # this is not yet included in the dataset
    # I0096_nimaigai2_senso is unannotated
    "sinatui": "I0096_sinatui", # this is not yet included in the dataset
    "nazuke": "I0097_kaicho_naa", # this is not yet included in the dataset
    "aisatsu": "I0139_aisacI", # this is not yet included in the dataset
    # I0307_masImuiuya_Va is unannotated
    "souchou_imo": "I0391_ssakai2", # this is not yet included in the dataset
    "kusakari": "I0391_ssakai", # this is not yet included in the dataset
    "hiroyuki_digital_museum": "I0396_museumaisatu_kocho", # this is not yet included in the dataset
    "ooura_bay": "I0398_ooura", # this is not yet included in the dataset
    "zyanmiga": "I0399_janmiga", # this is not yet included in the dataset
    "hidagaa": "I0400_hidagaa", # this is not yet included in the dataset
    "bubakariishi": "I0401_nintozeiseki", # this is not yet included in the dataset
    "tadashi_digital_museum": "I0402_kaichoaisatsu", # this is not yet included in the dataset
    "mitsuyoshi_digital_museum": "I0403_azachoaisatsu", # this is not yet included in the dataset
    # I0404_bituriiduu_Vl is unannotated
    "yamatugaa": "I0407_yamatugaa", # this is not yet included in the dataset
    "bissiutaki": "I0409b_bissi_utaki", # this is not yet included in the dataset
    "irabu_kankou3": "I0410_irabukankouVd", # this is not yet included in the dataset
    "irabu_kankou2": "I0410_irabukankouVc", # this is not yet included in the dataset
    "irabu_kankou1": "I0410_irabukankouVa", # this is not yet included in the dataset
    "mamegahana": "I0413_mamigahana_honban", # this is not yet included in the dataset
    "toujintorainouta": "I0415_toojintorai", # this is not yet included in the dataset
    # I0482_pic_wakimizu is unannotated
    "usu": "I0482_usI",
    "ngi": "I0482_ngi",
    "mutsuusa": "I0482_pic_mucIusa",
    "mancyuu": "I0482_manchu", # this is not yet included in the dataset
    "masugita": "I0482_masIgita", # this is not yet included in the dataset
    "maaninuhigi": "I0482_maaninuhigi", # this is not yet included in the dataset
    "maani": "I0482_maani", # this is not yet included in the dataset
    # I0482_ふじゃたてぃんぷら is unannotated
    "buuzu": "I0482_buuzI", # this is not yet included in the dataset
    "buugii": "I0482_pic_buugii",
    "kuwazuimo": "I0482_byuuigassa",
    "bippii": "I0482_bippii",
    "bikiyadumura": "I0482_bikiyadumura", # this is not yet included in the dataset
    "pii": "I0482_pii", # this is not yet included in the dataset
    "bantsugii": "I0482_bancIgii", # this is not yet included in the dataset
    "barazyan": "I0482_pic_barazan",
    "basa": "I0482_basa", # this is not yet included in the dataset
    "haka": "I0482_haka", # this is not yet included in the dataset
    "baaki": "I482_baaki", # this is not yet included in the dataset
    "niguu": "I0482_niguu",
    "takanna": "I0482_pic_takanna",
    "taka": "I0482_pic_taka",
    "suuni": "I0412_suuni",
    "satatinpura": "I0483_0414_satatinpura",
    "zzakugii": "I0482_pic_zzakugii",
    "kubazuu": "I0482_kubazII", # this is not yet included in the dataset
    "susuki": "I0482_gisIcI",
    "kami": "I0482_kami",
    "gazugii": "I0482_gazIhanagii",
    "ugan": "I0482_ugan",
    "oogomadara": "I0482_ayabasa", # this is not yet included in the dataset
    "adanbasaba": "I0482_adanbasaba", # this is not yet included in the dataset
    "yasaiitame": "I0483_0277_yasaiitame",
    # "butaniku_yasai_nimono": "I0483_0225_waatuhunya", # waatuhunya is lacking
    "butaniku_yasai_nimono": "I0483_0276_waa",
    "ishiusu": "I0483_0405_isIusI",
    "miyakosoba": "I0483_0527_suba",
    "waanimun": "I0483_0412_waanimun",
    "mamisuimai": "I0483_0688_mamisuimai",
    "fukyagi": "I0483_0409_fukyagi",
    "dakyau": "I0483_dakyau",
    "suzumebachi": "I0482_pc_taummabasI",
    "juusamai": "I0483_0528_zyuusamai", # this is not yet included in the dataset
    "saguna": "I0483_0678_saguna",
    "kkutsugii": "I0482_kkucIgii",
    "kayayaa": "I0482_kayayaa",
    "avvansu": "I0483_0222_avvansu", # I0483_うたき is unannotated
    "shiisaa": "I0488_しーさー", # cIccyu and guunya are lacking
    "aman": "I0490_あまん", # sabaucIgaa is lacking
    "ikemaoohashi": "I0490_池間大橋",
    "kaichou_shokureki_hiroyuki": "I0496_kaichosyokureki_hiroyuki",
    "kaichou_shokureki_tadashi": "I0496_kaichosyokureki_tadashi",
    "nakasunituimyaa": "I0501_nakasonetuimyaa",
    "harimizuutaki": "I0502_harimizuutaki",
    "nevsky": "I0503_Nevskynohi",
    "aisatsu_jikoshoukai1": "I0506_aisatsutehon_Va",
    "aisatsu_jikoshoukai2": "I0506_aisatsutehon_Vb",
    "aisatsu_jikoshoukai_hyoujun": "I0506_aisatsutehon_Vc",
    "aisatsu_otoori": "I0506_aisatsutehon_Vd",
    "aisatsu_roujinkai": "I0506_aisatsutehon_Ve",
    "aisatsu_yurinokai": "I0506_aisatsutehon_Vf",
    "mingu_huta": "I0507_mingukaisetsu_Va",
    "mingu_andira": "I0507_mingukaisetsu_Vb",
    "nakamautaki": "I0508_nakamautaki_V",
    "zyaagama": "I0509_zyaagama_V",
    "kyuukouminkan": "I0510_kyuukoominkan_Va",
    "kyuukouminkan_ura": "I0510_kyuukoominkan_Vb",
}


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
    ids = []
    
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
        # transcription = remove_tags(transcription) # no need to remove tags at this point; will be done later in training
        transcription = transcription.replace("んﾟ", "ん゚") # unify the devoiced nasal
        romaji = hiragana_to_romaji(transcription, romaji_mapping)
        phoneme = hiragana_to_phoneme(transcription, phoneme_mapping)
        
        # Identifier
        id = title_map.get(audio_name, audio_name)
        
        # Add the segment name, transcription, start, end, and audio to the data
        audios.append(segment_path)
        transcriptions.append(transcription)
        romajis.append(romaji)
        phonemes.append(phoneme)
        starts.append(start)
        ends.append(end)
        titles.append(audio_name)
        ids.append(id)
        
    audio_data = {
        "audio": audios,
        "transcription": transcriptions,
        "romaji": romajis,
        "phoneme": phonemes,
        "start": starts,
        "end": ends,
        "title": titles,
        "id": ids
    }
    
    return audio_data


def make_audio_data_from_splits_combining(audio: AudioSegment,
                                          annotations: List[Dict[str, str | int]],
                                          segments_subdir: str,
                                          audio_name: str,
                                          romaji_mapping: dict,
                                          phoneme_mapping: dict,
                                          max_duration: int) -> dict:
    """
    Create a dictionary of audio data from the splits with recursive augmentation by combining adjacent segments
    until the total duration reaches or exceeds max_duration.

    For each starting segment, the function:
      1. Saves the segment if its duration is less than max_duration.
      2. Recursively combines the segment with the next adjacent segments until the combined duration is >= max_duration.
      3. Saves every intermediate combined audio.
      4. Moves on to the next starting segment.

    Args:
        audio: The complete AudioSegment.
        annotations: A list of dictionaries, each containing 'start', 'end', and 'transcription' for one segment.
        segments_subdir: Directory where the exported audio segments will be saved.
        audio_name: Base name for the audio file segments.
        romaji_mapping: Mapping for converting Hiragana to Romaji.
        phoneme_mapping: Mapping for converting Hiragana to phoneme.
        max_duration: The maximum duration (in milliseconds) to reach when combining segments.
    
    Returns:
        A dictionary containing lists for audio paths, transcriptions, romaji, phoneme, start and end times, and titles.
    """
    audios = []
    transcriptions = []
    romajis = []
    phonemes = []
    starts = []
    ends = []
    titles = []
    ids = []
    
    # Assumes annotations are sorted by the 'start' time.
    n = len(annotations)
    for i in tqdm(range(n), desc="Processing segments"):
        # Start with the current segment as the base candidate.
        base_annotation = annotations[i]
        base_start = base_annotation["start"]
        base_end = base_annotation["end"]
        base_audio =  audio[base_start:base_end]
        base_transcription = remove_tags(base_annotation["transcription"])
        
        current_duration = base_end - base_start
        romaji = hiragana_to_romaji(base_transcription, romaji_mapping)  
        phoneme = hiragana_to_phoneme(base_transcription, phoneme_mapping)
        
        # Save the current candidate.
        segment_name = f"{audio_name}_{base_start}_{base_end}.wav"
        segment_path = os.path.join(segments_subdir, segment_name)
        base_audio.export(segment_path, format="wav")
        
        audios.append(segment_path)
        transcriptions.append(base_transcription)
        romajis.append(romaji)
        phonemes.append(phoneme)
        starts.append(base_start)
        ends.append(base_end)
        titles.append(audio_name)
        ids.append(title_map.get(audio_name, audio_name))
        
        # recursively combine with adjacent segments
        j = i + 1
        combined_transcription = base_transcription
        while j < n:
            # current_duration < max_duration and j < n:
            next_annotation = annotations[j]
            next_start = next_annotation["start"]
            next_end = next_annotation["end"]
            combined_duration = next_end - base_start
            if combined_duration >= max_duration:
                break
            
            combined_audio = audio[base_start:next_end]
            next_transcription = remove_tags(next_annotation["transcription"])
            combined_transcription += (" " + next_transcription)
            combined_romaji = hiragana_to_romaji(combined_transcription, romaji_mapping)
            combined_phoneme = hiragana_to_phoneme(combined_transcription, phoneme_mapping)
            
            combined_segment_name = f"{audio_name}_{base_start}_{next_end}.wav"
            combined_segment_path = os.path.join(segments_subdir, combined_segment_name)
            combined_audio.export(combined_segment_path, format="wav")
            
            audios.append(combined_segment_path)
            transcriptions.append(combined_transcription)
            romajis.append(combined_romaji)
            phonemes.append(combined_phoneme)
            starts.append(base_start)
            ends.append(next_end)
            titles.append(audio_name)
            ids.append(title_map.get(audio_name, audio_name))
            
            j += 1
        
    audio_data = {
        "audio": audios,
        "transcription": transcriptions,
        "romaji": romajis,
        "phoneme": phonemes,
        "start": starts,
        "end": ends,
        "title": titles,
        "id": ids
    }
    
    return audio_data


def prepare_format(dataset: dict,
                   audio,
                   segments_subdir: str,
                   audio_name: str,
                   romaji_mapping: dict,
                   phoneme_mapping: dict,
                   eaf_file: str,
                   transcription_tier: str = "sentence") -> dict:
    """Prepare the dataset format."""
    annotations = get_transcription(eaf_file,
                                    transcription_tier=transcription_tier)
    
    data = make_audio_data_from_splits(
        audio=audio,
        annotations=annotations,
        segments_subdir=segments_subdir,
        audio_name=audio_name,
        romaji_mapping=romaji_mapping,
        phoneme_mapping=phoneme_mapping
    )
    
    # combine the data
    dataset["audio"].extend(data["audio"])
    dataset["transcription"].extend(data["transcription"])
    dataset["romaji"].extend(data["romaji"])
    dataset["phoneme"].extend(data["phoneme"])
    dataset["start"].extend(data["start"])
    dataset["end"].extend(data["end"])
    dataset["title"].extend(data["title"])
    dataset["id"].extend(data["id"])
    print(f"Processed {audio_name} with {len(data['audio'])} segments.")
    return dataset


def generate_dataset(audio_dir: str,
                     romaji_map_file: str,
                     phoneme_map_file: str,
                     repo_name: str,
                     test_repo_name: str,
                     save_testdata_only: bool,
                     push_to_hub: bool,
                     validation_set: str = "jugon",
                     test_set: str = "mimamuibusu",
                     return_datasetdict: bool = False,
                     no_test: bool = False,
                     ) -> None | Dataset:
    """Generate a dataset from local wav/eaf files."""
    with open(romaji_map_file, "r") as f:
        romaji_mapping = json.load(f)
    with open(phoneme_map_file, "r") as f:
        phoneme_mapping = json.load(f)
        
    wav_eaf_pairs = get_wav_eaf_pairs(audio_dir)
    if not wav_eaf_pairs:
        print("No valid wav-eaf pairs found. Exiting.")
        exit(1)
    
    # Make a new dir for the segments    
    segments_dir = os.path.join(audio_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    
    dataset_dict = {
        "audio": [],
        "transcription": [],
        "romaji": [],
        "phoneme": [],
        "start": [],
        "end": [],
        "title": [],
        "id": []
    }
    
    valid_dataset_dict = {
        "audio": [],
        "transcription": [],
        "romaji": [],
        "phoneme": [],
        "start": [],
        "end": [],
        "title": [],
        "id": []
    }
    
    test_dataset_dict = {
        "audio": [],
        "transcription": [],
        "romaji": [],
        "phoneme": [],
        "start": [],
        "end": [],
        "title": [],
        "id": []
    }
    
    for pair in wav_eaf_pairs:
        wav_file = pair["wav"]
        eaf_file = pair["eaf"]
        
        audio = AudioSegment.from_wav(wav_file)
        audio_name = os.path.basename(wav_file)
        audio_name = os.path.splitext(audio_name)[0]
        
        if args.save_testdata_only and audio_name not in TEST_AUDIO:
            print(f"Skipping {audio_name} as it is not in the test audio list.")
            continue
        
        print(f"Processing {audio_name}...")
        
        segments_subdir = os.path.join(segments_dir, audio_name)
        os.makedirs(segments_subdir, exist_ok=True)
        
        # Get the transcription
        multispeaker_audios = {"kaichou_shokureki", "sinatui", "utakinohanashi", "nazuke"}
        if audio_name in multispeaker_audios:
            if audio_name == "kaichou_shokureki":
                target = ["tadashi", "hiroyuki"] # 0: tadashi, 1: hiroyuki
            elif audio_name == "sinatui" or audio_name == "utakinohanashi":
                target = ["obasan", "hiroyuki"] # 0: obasan, 1: hiroyuki
            elif audio_name == "nazuke":
                target = ["keichou", "hiroyuki"] # 0: keichou, 1: hiroyuki; order actually does not matter here
            else:
                raise NotImplementedError # TODO
            
            mono_audios = audio.split_to_mono()
            for i, role in enumerate(target):
                print(f"Processing {role}...")
                role_subdir = f"{args.audio_dir}/{audio_name}/{role}"
                os.makedirs(role_subdir, exist_ok=True)
                role_audio_file = mono_audios[i].export(
                    f"{role_subdir}/{audio_name}_{role}.wav",
                    format="wav"
                )
                role_audio = AudioSegment.from_wav(role_audio_file)
                annotations_role = get_transcription(
                    eaf_file,
                    transcription_tier=role
                )
                data_role = make_audio_data_from_splits(
                    audio=role_audio,
                    annotations=annotations_role,
                    segments_subdir=segments_subdir,
                    audio_name=f"{audio_name}_{role}",
                    romaji_mapping=romaji_mapping,
                    phoneme_mapping=phoneme_mapping
                )
                
                # Tests
                assert len(data_role["audio"]) == len(data_role["transcription"]), \
                    f"Audio and transcription lengths do not match for {audio_name}_{role}."
                assert len(data_role["audio"]) == len(data_role["start"]), \
                    f"Audio and start times lengths do not match for {audio_name}_{role}."
                assert len(data_role["audio"]) == len(data_role["end"]), \
                    f"Audio and end times lengths do not match for {audio_name}_{role}."
                assert len(data_role["audio"]) == len(data_role["title"]), \
                    f"Audio and titles lengths do not match for {audio_name}_{role}."
                    
                # Combine the data
                dataset_dict["audio"].extend(data_role["audio"])
                dataset_dict["transcription"].extend(data_role["transcription"])
                dataset_dict["romaji"].extend(data_role["romaji"])
                dataset_dict["phoneme"].extend(data_role["phoneme"])
                dataset_dict["start"].extend(data_role["start"])
                dataset_dict["end"].extend(data_role["end"])
                dataset_dict["title"].extend(data_role["title"])
                dataset_dict["id"].extend(data_role["id"])
        
        elif audio_name == validation_set:
            valid_dataset_dict = prepare_format(
                valid_dataset_dict,
                audio=audio,
                segments_subdir=segments_subdir,
                audio_name=audio_name,
                romaji_mapping=romaji_mapping,
                phoneme_mapping=phoneme_mapping,
                eaf_file=eaf_file,
                transcription_tier="sentence"
            )
        
        elif audio_name == test_set:
            test_dataset_dict = prepare_format(
                test_dataset_dict,
                audio=audio,
                segments_subdir=segments_subdir,
                audio_name=audio_name,
                romaji_mapping=romaji_mapping,
                phoneme_mapping=phoneme_mapping,
                eaf_file=eaf_file,
                transcription_tier="sentence"
            )
            
        else: # training data
            # Get the transcription
            transcription_tier = "default"
            annotations = get_transcription(eaf_file,
                                            transcription_tier=transcription_tier)
            if not annotations:
                print(f"No annotations found in {eaf_file}. Skipping.")
                continue
            
            if args.augment_combine:
                # Combine adjacent segments
                data = make_audio_data_from_splits_combining(
                    audio=audio,
                    annotations=annotations,
                    segments_subdir=segments_subdir,
                    audio_name=audio_name,
                    romaji_mapping=romaji_mapping,
                    phoneme_mapping=phoneme_mapping,
                    max_duration=15000  # 15 seconds
                )
            else:
                # Process segments separately
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
            dataset_dict["id"].extend(data["id"])
            
            print(f"Processed {audio_name} with {len(data['audio'])} segments.")
    
    # Create dataset splits
    train_dataset = Dataset.from_dict(dataset_dict).cast_column("audio", Audio(sampling_rate=16000))
    if no_test:
        datasetdict = DatasetDict({
            "train": train_dataset,
        })
    else:
        valid_dataset = Dataset.from_dict(valid_dataset_dict).cast_column("audio", Audio(sampling_rate=16000)) 
        test_dataset = Dataset.from_dict(test_dataset_dict).cast_column("audio", Audio(sampling_rate=16000))
        datasetdict = DatasetDict({
            "train": train_dataset,
            "validation": valid_dataset,
            "test": test_dataset
        })
    
    if return_datasetdict:
        return datasetdict
    
    if save_testdata_only:
        assert not no_test, "Cannot save test data when no_test is set."
        # Save the test dataset only
        test_dataset.save_to_disk(test_repo_name)
        print(f"Test dataset saved locally at: {test_repo_name}")
        if args.push_to_hub:
            test_dataset.push_to_hub(test_repo_name)
            print(f"Test dataset uploaded to Hugging Face Hub at: {test_repo_name}")
        exit(0)
        
    if push_to_hub:
        # Upload the dataset to Hugging Face Hub
        datasetdict.push_to_hub(repo_name)
        print(f"Dataset uploaded to Hugging Face Hub at: {repo_name}")
        
    else:
        # Save locally
        datasetdict.save_to_disk(repo_name)
    

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
    parser.add_argument(
        "--test_repo_name",
        type=str,
        default="ikema_youtube_asr_test",
        help="The name of the repository to use for uploading the test dataset to Hugging Face."
    )
    parser.add_argument(
        "--augment_combine",
        action='store_true',
        help="Augment the audio data by combining adjacent segments."
    )
    parser.add_argument(
        "--save_testdata_only",
        action='store_true',
        help="Only save the test dataset."
    )
    parser.add_argument(
        "--no_test",
        action='store_true',
        help="Do not create a test set."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    generate_dataset(audio_dir=args.audio_dir,
                     romaji_map_file=args.romaji_map_file,
                     phoneme_map_file=args.phoneme_map_file,
                     repo_name=args.repo_name,
                     test_repo_name=args.test_repo_name,
                     save_testdata_only=args.save_testdata_only,
                     push_to_hub=args.push_to_hub,
                     validation_set="jugon",
                     test_set="mimamuibusu",
                     return_datasetdict=False,
                     no_test=args.no_test
                     )
    