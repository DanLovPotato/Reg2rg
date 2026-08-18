# Smoke-test config: verify test_radgenome.py (inference) runs end-to-end
# on the 500-case validation subset. Uses the released pretrained weights
# as ckpt_path (not a freshly trained checkpoint) — this is only meant to
# confirm the inference code path works, not to measure model quality.

# Device settings — EDIT to match the server (run `nvidia-smi` there first)
cuda_devices="2"

# EDIT this one line to wherever you place data + checkpoints on the server;
# everything below is derived from it.
remote_base="/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT"

# Paths — mirrors the folder layout under chestCT/ here:
# llama_weights/, Reg2RG_weights/, smoke_data/reg2rg_data/dataset/...
lang_encoder_path="$remote_base/weights/llama_weights"
tokenizer_path="$remote_base/weights/llama_weights"
pretrained_visual_encoder="$remote_base/weights/Reg2RG_weights/RadFM_vit3d.pth"
pretrained_adapter="$remote_base/weights/Reg2RG_weights/RadFM_perceiver_fc.pth"
ckpt_path="$remote_base/outputs/Reg2RG_smoketest/checkpoint-10500/pytorch_model.bin"
data_folder="$remote_base/data/smoke_CTimagedata/reg2rg_data/dataset/valid_preprocessed"
mask_folder="$remote_base/data/smoke_CTimagedata/reg2rg_data/dataset/valid_region_mask"
report_file="$remote_base/data/smoke_CTimagedata/reg2rg_data/dataset/radgenome_files/validation_region_report.csv"
monai_cache_dir="$remote_base/data/smoke_CTimagedata/reg2rg_data/cache"
result_path="$remote_base/Reg2RG/results/Reg2RG_radgenome/inference_smoketest.csv"
