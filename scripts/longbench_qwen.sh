cd evaluation/LongBench

export CUDA_VISIBLE_DEVICES=2,3




model="Qwen2.5-32B-Instruct"

for budget in 512
do
    python -u pred_new.py \
        --model $model \
        --quest --token_budget $budget --chunk_size 16
done


python -m eval --model $model