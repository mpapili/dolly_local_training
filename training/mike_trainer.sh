#! /bin/bash
# TODO - shrank test-size down for a tiny input
# databricks/dolly-v2-3b
# EleutherAI/polyglot-ko-1.3b
# EleutherAI/pythia-410m-deduped
python3  \
     trainer.py \
     --input-model EleutherAI/polyglot-ko-1.3b \
     --deepspeed ../config/ds_z3_bf16_config.json \
     --epochs 2 \
     --local-output-dir ./training/outputdir \
     --dbfs-output-dir training/dbfs_output_dir \
     --per-device-train-batch-size 1 \
     --per-device-eval-batch-size 1 \
     --logging-steps 10 \
     --save-steps 200 \
     --save-total-limit 20 \
     --eval-steps 50 \
     --warmup-steps 0 \
     --test-size 5 \
     --lr 5e-6 \
     --bf16 false

