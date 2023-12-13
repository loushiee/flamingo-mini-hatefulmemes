# flamingo-mini-hatefulmemes

Multimodal visual-language model for [Hateful Memes](https://ai.meta.com/blog/hateful-memes-challenge-and-data-set/) dataset classification. This is based on [Flamingo-mini](https://github.com/dhansmair/flamingo-mini) source codes. Includes scripts for preprocessing the dataset images for blurring their texts. The OCR script is from https://github.com/HimariO/HatefulMemesChallenge/tree/main/data_utils.

How to perform training:
- place the `hateful_memes` dataset folder in the same folder as the `flamingo-mini-hatefulmemes` parent folder
- cd to `scripts` folder and run `run_ocr.sh` and `python preprocess_images.py`
- rename `hateful_memes/img_preprocesed` to `hateful_memes/img` (optionally move the original `hateful_memes/img` first)
- run `python setup.py install` to build the model
- cd to `training` folder and run `train.sh` to perform training
- cd to `training` folder and run `python test_hm.py` to perform test set evaluation

