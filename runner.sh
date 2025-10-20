
# python sample_prompts_from_mask.py \
#   --image /home/prithwijit/Vit/dog_image.jpg \
#   --mask /home/prithwijit/Vit/dog.jpg \
#   --density sparse \
#   --out prompts_sparse_dog.json

# python visualize_sam_cross_attention.py \
#   --image /home/prithwijit/Vit/texture_shape.png \
#   --prompts /home/prithwijit/Vit/attention/texture_shape.json \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --out /home/prithwijit/Vit/attention/segment-anything/cross_attn_results/t2s_sparse_cross.png \
#   --out-learned /home/prithwijit/Vit/attention/segment-anything/cross_attn_results/t2s_sparse_learned.png \
#   --out-i2t-all /home/prithwijit/Vit/attention/segment-anything/cross_attn_results/t2s_sparse_t2i.png \
#   --out-masks /home/prithwijit/Vit/attention/segment-anything/cross_attn_results/t2s_sparse_masks.png \


# python sam_encoder_attn_per_prompt.py \
#   --image /home/prithwijit/Vit/texture_shape.png \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --prompts /home/prithwijit/Vit/attention/texture_shape.json \
#   --out-dir /home/prithwijit/Vit/attention/segment-anything/sam_encoder_attn_per_prompt


# python analyze_prompt_influence.py \
#   --image /home/prithwijit/Vit/texture_shape.png \
#   --prompts /home/prithwijit/Vit/attention/texture_shape.json \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --radius 40 \
#   --out-dir /home/prithwijit/Vit/attention/segment-anything//prompt_influence



#   python mi_prompt_curves_plus.py \
#     --image /home/prithwijit/Vit/texture_shape.png \
#     --prompts /home/prithwijit/Vit/attention/texture_shape.json \
#     --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#     --model-type vit_h \
#     --out-dir /home/prithwijit/Vit/attention/segment-anything//mi_prompts_t2s \
#     --mine-steps 350 --infonce-steps 350 \
#     --seeds 4


#   python gaussian_normality_sam.py \
#     --image /home/prithwijit/Vit/dog_image.jpg \
#     --prompts /home/prithwijit/Vit/dog_0.json \
#     --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#     --model-type vit_h \
#     --out-dir /home/prithwijit/Vit/attention/segment-anything/gauss_check \
#     --block-idx -1 \
#     --n-proj 256 \
#     --alpha 0.05


#   python mi_prompt_curves_plus_gt.py \
#     --image /home/prithwijit/Vit/dog_image.jpg \
#     --prompts /home/prithwijit/Vit/dog_0.json \
#     --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#     --model-type vit_h \
#     --out-dir /home/prithwijit/Vit/attention/segment-anything/mi_prompts_gt1 \
#     --gt-mask /home/prithwijit/Vit/dog.jpg \
#     --mine-steps 200 --infonce-steps 200 --seeds 3

# python shapley_prompts_separate.py \
#   --image /home/prithwijit/Vit/dog_image.jpg \
#   --prompts /home/prithwijit/Vit/dog_0.json \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth --model-type vit_h \
#   --gt-mask /home/prithwijit/Vit/dog.jpg --perms 40 --mi-mode avg \
#   --out-dir /home/prithwijit/Vit/attention/segment-anything/shapley_sep


# python interactive_sam.py \
#   --image /home/prithwijit/Vit/attention/segment-anything/frog_snail_toad.jpg \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --out-dir ./outputs_time_series_frog_new


# python active_prompt_refine.py \
#   --image /home/prithwijit/Vit/attention/segment-anything/image_17.jpg \
#   --prompts /home/prithwijit/Vit/attention/segment-anything/outputs_masks/image_17_sampled_prompts.json \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --gt-mask /home/prithwijit/Vit/attention/segment-anything/shapley_sep/17_0_mask.png \
#   --gt-format png_nonzero_is_fg \
#   --out-dir /home/prithwijit/Vit/attention/segment-anything//active_refine_run_bird

python sam_from_prompts_once.py \
  --image /home/prithwijit/Vit/attention/segment-anything/frog_snail_toad.jpg \
  --prompts /home/prithwijit/Vit/attention/segment-anything/active_auto_frog/refined_prompts.json \
  --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --multimask \
  --gt-mask /home/prithwijit/Vit/attention/segment-anything/outputs_time_series_frog/frog_snail_toad_gt.npy \
  --out-dir /home/prithwijit/Vit/attention/segment-anything/sam_verify_frog1


# python sample_prompts_from_mask_new.py \
#   --image /home/prithwijit/Vit/attention/segment-anything/image_17.jpg \
#   --mask /home/prithwijit/Vit/attention/segment-anything/shapley_sep/17_0_mask.png \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --num-pairs 50 \
#   --out-dir ./outputs_masks

# python auto_prompt_suggester.py \
#   --image /home/prithwijit/Vit/attention/segment-anything/frog_snail_toad.jpg \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --prompts /home/prithwijit/Vit/attention/segment-anything/outputs_time_series_frog1/frog_snail_toad_prompts.json \
#   --gt-mask /home/prithwijit/Vit/attention/segment-anything/outputs_time_series_frog/frog_snail_toad_gt.npy --gt-format npy_one_is_fg \
#   --out-dir /home/prithwijit/Vit/attention/segment-anything/auto_suggest \
#   --use-tta --save-explain

# python auto_prompt_suggester.py \
#   --image /home/prithwijit/Vit/attention/segment-anything/frog_snail_toad.jpg \
#   --prompts /home/prithwijit/Vit/attention/segment-anything/outputs_time_series_frog_new/frog_snail_toad_prompts.json \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --gt-mask /home/prithwijit/Vit/attention/segment-anything/outputs_time_series_frog/frog_snail_toad_gt.npy \
#   --gt-format npy_one_is_fg \
#   --out-dir /home/prithwijit/Vit/attention/segment-anything/active_auto_frog \
#   --use-tta \
#   --save-explain
