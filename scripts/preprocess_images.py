import mmcv
import matplotlib.pyplot as plt
import glob
import os
from mmagic.apis import MMagicInferencer

editor = MMagicInferencer('global_local', model_setting=1)

hateful_memes_path  = os.path.join('..', '..', 'hateful_memes')
masks_path = os.path.join(hateful_memes_path, "img_mask_3px")
output_path = os.path.join(hateful_memes_path, "img_preprocessed")
if not os.path.exists(output_path):
    os.mkdir(output_path) 

iglob_pattern = os.path.join(masks_path, "*[0-9].png")
print(f"Processing {len(list(glob.iglob(iglob_pattern)))} files from {iglob_pattern}")

count = 0
for img in glob.iglob(iglob_pattern):
    _, filename = os.path.split(img)
    name = filename.split('.')[0]
    mask = os.path.join(masks_path, f"{name}.mask.png")
    temp_img = '.img.png'
    temp_mask = '.mask.png'
    mmcv.imwrite(mmcv.impad_to_multiple(mmcv.imread(img), 8), temp_img)
    mmcv.imwrite(mmcv.impad_to_multiple(mmcv.imread(mask), 8), temp_mask)
    img_out = os.path.join(output_path, f"{name}.png")
    _ = editor.infer(img=temp_img, mask=temp_mask, result_out_dir=img_out)

    # print(f"{count}: name: {name} output: '{img_out}'")
    count += 1
    if count % 100 == 0:
        print(f"Processed {count} images")

print("DONE")
