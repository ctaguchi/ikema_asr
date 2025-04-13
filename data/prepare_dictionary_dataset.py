import datasets
import os
import glob
import pandas as pd
import argparse

def read_dictionary(dict_path: str) -> pd.DataFrame:
    """Read the dictionary file."""
    orig_df = pd.read_csv(dict_path,
                          sep="\t")
    
    # Remove unnecessary columns
    df = orig_df[
        ["WordID", "Wd", "SoundFile", "PoS", "Description", "ExampleSentence"]
    ]
    """
    Example:
    	WordID	Wd	SoundFile	PoS	Description
        0	1	あ	ikmvoc_1.wav	助詞	提題を表す、「は」。提題を表す「あ」はつく語の最後の音に従って形が変わる（１）aで終わる語の...
        1	2	あ	ikmvoc_2.wav	助詞	目的語を表す（「う」に対して、第二目的格ともよばれる）、「を」。提題を表す「あ」と同じ変化をする
        2	3	あー	ikmvoc_3.wav	名詞	粟
    """
    return df


def filter_df(df: pd.DataFrame,
              audio_dir: str) -> pd.DataFrame:
    """Add some modification to the df if needed."""
    # Sort the dataset by 'WordID' to ensure consistent ordering
    df = df.sort_values(by="WordID").reset_index(drop=True)
    
    # Filter out rows where the SoundFile does not exist in the audio directory
    sound_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    sound_file_names = [os.path.basename(file) for file in sound_files]
    df = df[df["SoundFile"].isin(sound_file_names)]
    
    # Fix the inconsistencies of the sound file names and SoundFile
    # From WordID 5272 to 5484, the correct sound file is "ikmvoc_{next_word_id}.wav"
    mask = (df['WordID'] >= 5272) & (df['WordID'] <= 5484 + 1)
    word_ids = df.loc[mask, 'WordID'].values
    
    # Ensure we don't go out of bounds when accessing the next WordID
    for i in range(len(word_ids) - 1):
        current_index = df[df['WordID'] == word_ids[i]].index[0]
        next_word_id = word_ids[i + 1]  # The WordID of the next row in sorted order
        df.at[current_index, 'SoundFile'] = f"ikmvoc_{next_word_id}.wav"

    # Make WordID=5485's SoundFile to NA
    df.loc[df['WordID'] == 5485, 'SoundFile'] = None
        
    # Filter out rows with missing values in 'Wd' or 'SoundFile'
    df = df.dropna(subset=["Wd", "SoundFile"])
        
    return df


def make_hf_dataset(df: pd.DataFrame,
                    audio_dir: str) -> list:
    """Create a dataset from the dataframe."""
    wordid_list = df["WordID"].tolist()
    word_list = df["Wd"].tolist()
    soundfile_list = df["SoundFile"].tolist()
    pos_list = df["PoS"].tolist()
    description_list = df["Description"].tolist()
    example_sentence_list = df["ExampleSentence"].tolist()
    
    soundfile_list = [os.path.join(audio_dir, f) if f is not None else None
                      for f in soundfile_list]
    example_sentence_list = [x if isinstance(x, str) else "" for x in example_sentence_list]
    
    # Test if all list items are str or int
    assert all(isinstance(x, (str, type(None))) for x in soundfile_list), \
        "All items in the 'SoundFile' list should be strings or None. " \
        f"Found: {soundfile_list}."
    assert all(isinstance(x, str) for x in word_list), \
        "All items in the 'Wd' list should be strings. " \
        f"Found: {word_list}."
    assert all(isinstance(x, str) for x in pos_list), \
        "All items in the 'PoS' list should be strings. " \
        f"Found: {pos_list}."
    assert all(isinstance(x, str) for x in description_list), \
        "All items in the 'Description' list should be strings. " \
        f"Found: {description_list}."
    for x in example_sentence_list:
        if not isinstance(x, str):
            print(x)
            print(type(x))
            print(x is None)
            raise ValueError(
                f"All items in the 'ExampleSentence' list should be strings. "
                f"Found an invalid type: {type(x)} for value: {x}. "
                "Please ensure all entries are strings."
            )
    
    
    dataset_dict = {
        "word_id": wordid_list,
        "word": word_list,
        "audio": soundfile_list,
        "part_of_speech": pos_list,
        "description": description_list,
        "example_sentence": example_sentence_list
    }
    
    # Create Hugging Face Dataset
    dataset = datasets.Dataset.from_dict(dataset_dict)

    # Cast the 'SoundFile' column to Audio format
    dataset = dataset.cast_column("audio",
                                  datasets.Audio(sampling_rate=16000))
    
    return dataset


def get_args() -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare a dictionary dataset."
    )
    parser.add_argument(
        "--dict_path",
        type=str,
        default="./dictionary/ikema20241005.txt",
        help="Path to the dictionary file"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="./dictionary/ikmvoc",
        help=(
            "Path to the directory containing audio files. "
            "This should contain the sound files referenced in the dictionary."
        )
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
        default="ikema_dict_asr",
        help=(
            "The name of the repository to use for uploading the dataset to Hugging Face. "
            "This is only used if `--upload_to_hf` is set. "
            "Make sure to create the repo on Hugging Face before uploading."
        )
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    print("Reading the dictionary data...")
    df = read_dictionary(args.dict_path)
    print("Dictionary read.")
    
    print("Filtering the DataFrame based on audio files...")
    df = filter_df(df, args.audio_dir)
    print(f"Filtered DataFrame contains {len(df)} entries after filtering.")
    
    print("Creating Hugging Face dataset...")
    dataset = make_hf_dataset(
        df,
        args.audio_dir
    )
    print("Hugging Face dataset created.")
    
    if args.push_to_hub:
        # Upload the dataset to Hugging Face Hub
        dataset.push_to_hub(args.repo_name)
        print(f"Dataset uploaded to Hugging Face Hub at: {args.repo_name}")
    else:
        # Save locally
        dataset.save_to_disk(args.repo_name)
        print(f"Dataset saved locally at: {args.repo_name}")