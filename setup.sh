ssh ubuntu@147.185.41.15
cd ~/TensoIRExp
source tensoIR/bin/activate



# 先只用一个场景测试
mkdir -p /tmp/test_meta
cp data_samples/relight_metadata/marble_bust_01_env_2.json /tmp/test_meta/

python scripts/train_and_relight_polyhaven.py \
    --config configs/single_light/polyhaven_lvsm.txt \
    --data_root /data/polyhaven_lvsm/test \
    --relight_meta_dir /tmp/test_meta \
    --output_dir ./output/polyhaven_relight_test \
    --vis_every 5000 \
    --n_iters 30000
