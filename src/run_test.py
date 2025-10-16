from transformers import AutoProcessor, AutoModelForCTC
import torch
import jiwer
import argparse
from tqdm import trange
import json
import os

# local imports
from finetune import load_data

MODEL_NAME = "ikema-asr-indomain-ph"
MAIN_DATA = "ikema_youtube_asr"
DICT_DATA = "ikema_dict_asr"
STORY_DATA = "ikema_youtube_asr_test"
LECTURE_DATA = "ikema_youtube_asr_hougenkougi"
NUM_FRAMES_TO_CHECK = 10


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="test mode."
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help="Model name or path."
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference."
    )
    return parser.parse_args()


def predict(batch):
    # Prepare input features
    arrays = [x["array"] for x in batch["audio"]]
    sampling_rate = batch["audio"][0]["sampling_rate"]
    
    inputs = processor(
        arrays,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["predicted_text"] = processor.batch_decode(pred_ids,
                                                     skip_special_tokens=True)
    return batch


if __name__ == "__main__":
    args = get_args()

    print("Loading data...")
    dataset_dict = load_data(
        main_dataset=MAIN_DATA,
        eval_data=False,
        load_from_disk=False,
        use_dict_dataset=True,
        story_dataset=STORY_DATA,
        lecture_dataset=LECTURE_DATA,
        dict_dataset=DICT_DATA
        )

    dev = dataset_dict["dev"]
    test = dataset_dict["test"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Used device: {device}")
    print("Loading the model...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForCTC.from_pretrained(args.model_name).to(device)
    print("Model and processor loaded.")

    # TODO: The vocab has "[PAD]" and "[UNK]", which are added manually when training,
    # but also it seems to have "<pad>" and "<unk>" which seem to have been added
    # after the vocab is created, probably when processor is defined. This might be a
    # change after an update in transformers.
    # As a temporary remedy, I'm forcing the processor to have "[PAD]" as the padding token.
    processor.tokenizer.pad_token = "[PAD]"
    processor.tokenizer.pad_token_id = processor.tokenizer.convert_tokens_to_ids("[PAD]")
    
    model.eval()

    if args.test:
        results = test.map(predict,
                           batched=True,
                           batch_size=args.batch_size)
        # metrics
        print("Inference done.")
        print("Computing the metrics...")
        if args.model_name.endswith("romaji-ph"):
            refs = results["romaji"]
        else:
            refs = results["transcription"]
        preds = results["predicted_text"]
        
        wer = jiwer.wer(refs, preds)
        cer = jiwer.cer(refs, preds)

        print(f"WER: {wer}")
        print(f"CER: {cer}")

        results = {
            "wer": wer,
            "cer": cer,
            "preds": preds,
            "refs": refs
        }

        results_file = os.path.join(args.model_name, "metrics.json")
        with open(results_file, "w") as f:
            json.dump(results,
                      f,
                      ensure_ascii=False,
                      indent=4)
        print(f"Results saved to {results_file}")
        
    else:
        # Optionally: inspect vocab for context
        print("Inspecting the vocab...")
        try:
            vocab_tokens = processor.tokenizer.convert_ids_to_tokens(range(len(processor.tokenizer)))
            print("\nVocab size:", len(vocab_tokens))
            print("Sample vocab entries:", vocab_tokens)
        except Exception as e:
            print("\nCould not access tokenizer vocabulary:", e)

        print("Special tokens:")
        print(processor.tokenizer.special_tokens_map)
        print("Pad token:", processor.tokenizer.pad_token)
        print("Pad token ID:", processor.tokenizer.pad_token_id)

        print("Inspecting the output...")
        for i in range(10):
            print(f"Dev sample No. {i}")
            audio = dev[i]["audio"]
            waveform = torch.tensor(audio["array"]).unsqueeze(0)
            sr = audio["sampling_rate"]
        
            inputs = processor(
                waveform.squeeze().numpy(),
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )
            input_values = inputs.input_values.to(device)

            with torch.no_grad():
                logits = model(input_values).logits  # [B, T, V]
            pred_ids = torch.argmax(logits, dim=-1)

            # debug
            batch_index = 0
            last_ids = pred_ids[batch_index, -NUM_FRAMES_TO_CHECK:]
            tokens = [processor.tokenizer.decode([i]) for i in last_ids.tolist()]

            print(f"=== Debugging sample #{i} ===")
            print("Last frame token IDs:")
            print(last_ids.tolist())
            print("\nDecoded tokens for last frames:")
            print(tokens)
            
            decoded_text = processor.batch_decode(pred_ids)[0]
            print("\nDecoded text:")
            print(decoded_text)
            pred = processor.batch_decode(pred_ids,
                                          skip_special_tokens=True)[0]
            print("Prediction text:")
            print(pred)
            print("Reference text:")
            print(dev[i].get("transcription", "<no transcript field>"))
            

    print("DONE.")
