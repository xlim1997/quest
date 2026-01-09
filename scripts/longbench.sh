cd evaluation/LongBench


export CUDA_VISIBLE_DEVICES=1
model="Meta-Llama-3.1-8B-Instruct"
# #full kv 
# python -u pred.py --model $model 

for budget in 512
do
    python -u pred.py \
        --model $model \
done


python -u eval.py --model $model