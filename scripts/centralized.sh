for seed in {0..9}
do
  python centralized_train.py --exp_name centralized --seed $seed --model CNN
done