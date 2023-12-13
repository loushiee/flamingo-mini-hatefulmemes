SELF=$(dirname "$(realpath $0)")
MEME_ROOT_DIR="$SELF/../../hateful_memes"
echo "DBGR MEME_ROOT_DIR: $MEME_ROOT_DIR"

# OCR to get text bbox and mask
echo "[OCR] detect start"
python3 ocr.py detect "$MEME_ROOT_DIR"
echo "[OCR] convert point annotation to box"
python3 ocr.py point_to_box "$MEME_ROOT_DIR/ocr.json"
echo "[OCR] create text segmentation mask"
python3 ocr.py generate_mask "$MEME_ROOT_DIR/ocr.box.json" "$MEME_ROOT_DIR/img" "$MEME_ROOT_DIR/img_mask_3px"
