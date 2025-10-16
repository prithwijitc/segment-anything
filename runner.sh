
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


  python gaussian_normality_sam.py \
    --image /home/prithwijit/Vit/dog_image.jpg \
    --prompts /home/prithwijit/Vit/dog_0.json \
    --checkpoint /home/prithwijit/Vit/sam_vit_h_4b8939.pth \
    --model-type vit_h \
    --out-dir /home/prithwijit/Vit/attention/segment-anything/gauss_check \
    --block-idx -1 \
    --n-proj 256 \
    --alpha 0.05
