
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
#   --out-dir ./outputs_time_series_frog_new2


# python active_prompt_refine.py \
#   --image /home/prithwijit/Vit/attention/segment-anything/image_17.jpg \
#   --prompts /home/prithwijit/Vit/attention/segment-anything/outputs_masks/image_17_sampled_prompts.json \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --out-dir /home/prithwijit/Vit/attention/segment-anything/active_refine_run_bird_with_gt

# python sam_from_prompts_once.py \
#   --image /home/prithwijit/Vit/attention/segment-anything/image_17.jpg \
#   --prompts /home/prithwijit/Vit/attention/segment-anything/outputs_masks/image_17_sampled_prompts.json \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --multimask \
#   --gt-mask /home/prithwijit/Vit/attention/segment-anything/outputs_time_series_frog/frog_snail_toad_gt.npy \
#   --out-dir /home/prithwijit/Vit/attention/segment-anything/sam_verify_bird1


# python sample_prompts_from_mask_new.py \
#   --image /home/prithwijit/Vit/attention/segment-anything/frog_snail_toad.jpg \
#   --mask /home/prithwijit/Vit/attention/segment-anything/outputs_time_series_frog/frog_snail_toad_gt.npy \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --num-pairs 30 \
#   --out-dir ./outputs_masks_frog

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
#   --prompts /home/prithwijit/Vit/attention/segment-anything/outputs_time_series_frog_new1/frog_snail_toad_prompts.json \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --out-dir /home/prithwijit/Vit/attention/segment-anything/active_auto_frog2 \
#   --use-tta \
#   --save-explain

# python auto_prompt_suggester.py \
#   --image /home/prithwijit/Vit/attention/segment-anything/image_17.jpg \
#   --prompts /home/prithwijit/Vit/attention/segment-anything/outputs_masks/image_17_sampled_prompts.json \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --out-dir /home/prithwijit/Vit/attention/segment-anything/active_auto_bird1 \
#   --use-tta \
#   --save-explain

# python interactive_sam_suggestor.py \
#   --image /home/prithwijit/Vit/attention/segment-anything/frog_snail_toad.jpg \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --out-dir /home/prithwijit/Vit/attention/segment-anything/interactive_demo

# MPLBACKEND=Qt5Agg python interactive_active_prompting.py \
#   --image /home/prithwijit/Vit/attention/segment-anything/frog_snail_toad.jpg \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --out-dir /home/prithwijit/Vit/attention/segment-anything/out_interactive \
#   --use-tta


  # python prompt_mi_curve.py \
  #   --image /home/prithwijit/Vit/attention/segment-anything/image_17.jpg \
  #   --gt /home/prithwijit/Vit/attention/segment-anything/shapley_sep/17_0_mask.png \
  #   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
  #   --model-type vit_h \
  #   --iterations 100 \
  #   --out-dir /home/prithwijit/Vit/attention/segment-anything/runs/demo1


    # python prompt_mi_cmi_curve.py \
    # --image /home/prithwijit/Vit/attention/segment-anything/image_17.jpg \
    # --gt /home/prithwijit/Vit/attention/segment-anything/shapley_sep/17_0_mask.png \
    # --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
    # --model-type vit_h \
    # --iterations 100 \
    # --samples-per-iter 16 \
    # --out-dir /home/prithwijit/Vit/attention/segment-anything/runs/demo2 \



    #   python make_prompts_strict.py \
    # --image /home/prithwijit/Vit/attention/segment-anything/image_17.jpg \
    # --gt /home/prithwijit/Vit/attention/segment-anything/shapley_sep/17_0_mask.png \
    # --pairs 20 \
    # --num-files 220 \
    # --calib-count 200 \
    # --out-dir /home/prithwijit/Vit/attention/segment-anything/out_prompts_strict_bird \
    # --seed 0

  # python robust_calibrate_T.py \
  # --image /home/prithwijit/Vit/attention/segment-anything/image_17.jpg \
  # --gt /home/prithwijit/Vit/attention/segment-anything/shapley_sep/17_0_mask.png \
  # --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
  # --model-type vit_h \
  # --calib-dir /home/prithwijit/Vit/attention/segment-anything/out_prompts_strict_bird/calib \
  # --eval-dir  /home/prithwijit/Vit/attention/segment-anything/out_prompts_strict_bird/eval \
  # --multimask \
  # --T-lo 0.3 --T-hi 3.0 --grid 61 --rounds 3 \
  # --out-dir /home/prithwijit/Vit/attention/segment-anything/runs/calib_one_robust \
  # --seed 0

# python mc_sufficiency_refine.py \
#   --image /home/prithwijit/Vit/attention/segment-anything/image_17.jpg \
#   --gt /home/prithwijit/Vit/attention/segment-anything/shapley_sep/17_0_mask.png \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --eval-dir /home/prithwijit/Vit/attention/segment-anything/out_prompts_strict_bird/eval \
#   --calib-json /home/prithwijit/Vit/attention/segment-anything/runs/calib_one_robust/calibration.json \
#   --multimask \
#   --mc-samples 64 \
#   --jitter-radius 2 \
#   --ce-eps 1e-4 \
#   --ce-tol 1e-4 \
#   --miou-tol 0.0 \
#   --out-dir /home/prithwijit/Vit/attention/segment-anything/runs/mc_sufficiency_one \
#   --seed 0


# python mc_sufficiency_base_eval.py \
#   --image /home/prithwijit/Vit/attention/segment-anything/image_17.jpg \
#   --gt /home/prithwijit/Vit/attention/segment-anything/shapley_sep/17_0_mask.png \
#   --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
#   --model-type vit_h \
#   --eval-dir /home/prithwijit/Vit/attention/segment-anything/out_prompts_strict_bird/eval \
#   --calib-json /home/prithwijit/Vit/attention/segment-anything/runs/calib_one_robust/calibration.json \
#   --multimask \
#   --mc-samples 64 \
#   --jitter-radius 2 \
#   --ce-eps 1e-4 \
#   --ce-tol 1e-4 \
#   --miou-tol 0.0 \
#   --out-dir /home/prithwijit/Vit/attention/segment-anything/runs/mc_sufficiency_base_eval \
#   --seed 0

python gtfree_minmax_refiner.py \
  --image /home/prithwijit/Vit/attention/segment-anything/image_17.jpg \
  --prompts /home/prithwijit/Vit/attention/segment-anything/outputs_masks/image_17_sampled_prompts.json \
  --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --out-dir /home/prithwijit/Vit/attention/segment-anything/runs/minmax \
  --multimask \
  --tta \
  --mc-samples 48 \
  --env-samples 96 \
  --jitter-radius 2 \
  --trim-frac 0.1 \
  --eps-mode full \
  --eps-scale 1.00 \
  --plateau 1e-4 \
  --seed 0 \
  --progress



