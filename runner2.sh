# python /home/prithwijit/Vit/attention/segment-anything/gtfree_minmax_refiner_fast.py \
#   --image /home/prithwijit/Vit/attention/segment-anything/image_17.jpg \
#   --prompts /home/prithwijit/Vit/attention/segment-anything/outputs_masks/image_17_sampled_prompts.json \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --out-dir /home/prithwijit/Vit/attention/segment-anything/runs/minmax_fast \
#   --multimask \
#   --tta \
#   --mc-samples 48 \
#   --env-samples 96 \
#   --jitter-radius 2 \
#   --trim-frac 0.1 \
#   --eps-mode full \
#   --eps-scale 1.00 \
#   --plateau 1e-4 \
#   --seed 0 \
#   --progress


# python /home/prithwijit/Vit/attention/segment-anything/refiner.py   \
#   --image /home/prithwijit/Vit/attention/segment-anything/image_17.jpg \
#   --prompts /home/prithwijit/Vit/attention/segment-anything/outputs_masks/image_17_sampled_prompts.json \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --multimask --tta --progress


# python /home/prithwijit/Vit/attention/segment-anything/refine_overlay.py   \
#     --image /home/prithwijit/Vit/attention/segment-anything/image_17.jpg \
#     --prompts /home/prithwijit/Vit/attention/segment-anything/outputs_masks/image_17_sampled_prompts.json \
#     --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#     --model-type vit_h \
#     --multimask --tta --progress


# python gt_free_part_aware_refiner_improved.py \
#   --image /home/prithwijit/Vit/attention/segment-anything/image_17.jpg \
#   --prompts /home/prithwijit/Vit/attention/segment-anything/outputs_masks/image_17_sampled_prompts.json \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --out-dir runs/gtfree_refiner_robust \
#   --calibrate --tta --progress \
#   --env-samples 128 --mc-samples 96 --bootstrap 600 \
#   --strata 0.25,0.5,0.75 \
#   --rpos 1 --rneg 1 \
#   --group-margin 0.10 --group-minlink 0.8 --sp-segments 1200



# Example
python auto_prompter_experiment.py \
  --images-root /data/prithwijit/vit-attn/pointprompt/Images \
  --masks-root  /data/prithwijit/vit-attn/pointprompt/Masks \
  --out-dir     /home/prithwijit/Vit/attention/segment-anything/auto_prompt_results \
  --checkpoint  /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
  --model-type  vit_h \
  --iterations  100 \
  --normalize-to max \
  --normalization-metric miou

