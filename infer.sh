VLM_API_KEY="$(cat ~/.config/openrouter/api_key)" \
VLM_PROXY=http://127.0.0.1:3128 \
python inference_grounded.py \
  --root-model-dir /mnt/data1/weights/FlowDIS \
  --image-path assets/Playarena-Indoor.jpg \
  --prompt "the pedal kart ridden by the kid wearing a cyan T-shirt" \
  --output-path assets/disambiguate/mask.png \
  --grounding-path assets/disambiguate/grounding.json \
  --composite-path assets/disambiguate/cutout.png \
  --greenscreen-path assets/disambiguate/greenscreen.png \
  --overlay-path assets/disambiguate/overlay.png \
  --grounding-result-path assets/disambiguate/grounding_result.png \
  --int8