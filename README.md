# Ikema ASR
This repository contains the experimental code used to develop the ASR model for the Ikema Miyakoan language presented at LREC 2026.

## Datasets
- [Field dataset](https://huggingface.co/datasets/ctaguchi/ikema_youtube_asr_full)
- [Audiobook dataset](https://huggingface.co/datasets/ctaguchi/ikema_youtube_asr_test)
- [Dictionary dataset](https://huggingface.co/datasets/ctaguchi/ikema_dict_asr)

The Dictionary dataset is separated because its table columns are different from the other two.
Please refer to the experimental code as to how to combine the datasets.

## Models
- Model: [300M parameters](https://huggingface.co/ctaguchi/ikema-asr-indomain-ph)

## License
The datasets are available under the CC SA-4.0 license.
The code is released under the MIT license.

## Citation
TBA (to be presented at LREC2026. We also plan to upload it to arXiv.)