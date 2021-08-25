python3 ./.v2e/v2e.py -i ./.v2e/input/tennis.mov -o ./.v2e/output \
  --avi_frame_rate=120 --overwrite --auto_timestamp_resolution \
  --output_height 720 --output_width 1280  --dvs_params clean --pos_thres=.15 --neg_thres=.15 \
  --dvs_emulator_seed=0 --slomo_model=./.v2e/.pretrain/SuperSloMo39.ckpt --no_preview \
  --dvs_text=./output