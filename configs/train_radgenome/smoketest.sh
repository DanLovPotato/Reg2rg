# Smoke-test config: verify train_radgenome.py runs end-to-end on the
# 500-case validation subset (used here as a stand-in for train data —
# this matches the ModelArguments/DataArguments defaults in
# src/args/train_radgenome/jhcpu7.py, which also point at the "valid_*" files).

# Experiment settings
experiment_name="Reg2RG_smoketest"
bf16=True

# Device settings — EDIT to match the server (run `nvidia-smi` there first)
cuda_devices="0,1"

# Torchrun settings
master_port=25368

# EDIT this one line to wherever you place data + checkpoints on the server;
# everything below is derived from it.
remote_base="/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT"

# Paths — mirrors the folder layout under chestCT/ here:
# llama_weights/, Reg2RG_weights/, smoke_data/reg2rg_data/dataset/...
lang_encoder_path="$remote_base/llama_weights"
tokenizer_path="$remote_base/llama_weights"
pretrained_visual_encoder="$remote_base/Reg2RG_weights/RadFM_vit3d.pth"
pretrained_adapter="$remote_base/Reg2RG_weights/RadFM_perceiver_fc.pth"
data_folder="$remote_base/smoke_data/reg2rg_data/dataset/valid_preprocessed"
mask_folder="$remote_base/smoke_data/reg2rg_data/dataset/valid_region_mask"
report_file="$remote_base/smoke_data/reg2rg_data/dataset/radgenome_files/validation_region_report.csv"
monai_cache_dir="$remote_base/smoke_data/reg2rg_data/cache"
output_dir="$remote_base/outputs/$experiment_name"
deepspeed_config="../ds_configs/stage2.json"

# Training settings — kept minimal for a fast smoke test, not a real training run
learning_rate=5e-5
per_device_train_batch_size=1
num_train_epochs=200
gradient_accumulation_steps=1
evaluation_strategy="no"
save_strategy="epoch"
save_total_limit=1
weight_decay=0.0
warmup_steps=0
lr_scheduler_type="constant_with_warmup"
dataloader_num_workers=4
logging_steps=1
