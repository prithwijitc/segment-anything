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


python /home/prithwijit/Vit/attention/segment-anything/refine_overlay.py   \
    --image /home/prithwijit/Vit/attention/segment-anything/image_17.jpg \
    --prompts /home/prithwijit/Vit/attention/segment-anything/outputs_masks/image_17_sampled_prompts.json \
    --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
    --model-type vit_h \
    --multimask --tta --progress
