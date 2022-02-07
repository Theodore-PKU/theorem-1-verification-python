## Dataset

### ImageNet

We downloaded ImageNet from https://image-net.org. The link is https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar. In our experiment, we only use training set.

We use the following shell script to extract the `.tar` files.

```shell
for file in *.tar; do tar -xf "$file"; rm "$file"; done
```

We save the images in `../datasets/imagenet/train`. This directory is not in this repo.



### MNIST

We downloaded MNIST trough pytorch. We provided training and testing sets in `/data/mnist` which are saved as `/data/mnist/training.pt` and `/data/mnist/test.pt`. In the directory of `/data/mnist`, there are some `.pkl` files which save the noisy labels for different types of noise at different noise levels.



## DDPM-SR model

The DDPM-SR model used in our experiment can be downloaded from https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64_256_upsampler.pt. It should be saved in the directory of `/checkpoints/ddpm_sr`. Since it is a big file, we did not provided in this repo.



## Experiments

### MNIST

- Generating noisy labels and training models are all executed by `mnist_classification.py`. The different tasks depend on the args. 

```shell
python mnist_classification.py  # args are neglected.
```

In this experiments, "uniform", "fix", and "model" in the code mean uniform noise, biased noise, and generated noise in our paper respectively.



### ImageNet

- Generate sr samples using ddpm-sr model.

```shell
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --large_size 256 --small_size 64 \
--num_channels 192 --num_heads 4 --num_res_blocks 2 --learn_sigma True --resblock_updown True \
--use_fp16 False --use_scale_shift_norm True"
GD_FLAGS="--diffusion_steps 1000 --noise_schedule linear --learn_sigma True --timestep_respacing 250"
SCRIPT_FLAGS="--log_dir logs/ddpm_sr/64_256_step_250 --output_dir outputs/ddpm_sr/64_256_step_250 \
--data_dir ../datasets/imagenet/train --model_path checkpoints/ddpm_sr/64_256_upsampler.pt \
--to_sample_images_dict_path data/sr_imagenet/to_sample_data_info.pkl \
--num_samples_per_image 100 --batch_size 10"

python ddpm_sr_sample.py $MODEL_FLAGS $GD_FLAGS $SCRIPT_FLAGS
```

Generated samples will be saved in the directory of `/outputs/ddpm_sr/64_256_step_250 `.



- Training the mean model.

```shell
MODEL_FLAGS="--attention_resolutions 16 --large_size 256 --small_size 64 \
--model_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True \
--use_fp16 True --channel_mult 1,1,2,2,4"
SCRIPT_FLAGS="--data_dir ../datasets/imagenet/train --data_info_dict_path data/sr_imagenet/train_data_info.pkl \
--batch_size 64 --microbatch 8 \
--log_dir logs/sr_mean/64_256 --log_interval 10 --save_interval 5000 \
--model_save_dir checkpoints/sr_mean/64_256 \
--debug_mode yes"

python sr_mean_train.py $MODEL_FLAGS $SCRIPT_FLAGS
```

Trained models (checkpoints) will saved in the directory of `checkpoints/sr_mean/64_256`.



- Testing the mean model.

```shell
MODEL_FLAGS="--attention_resolutions 16 --large_size 256 --small_size 64 \
--model_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True \
--use_fp16 True --channel_mult 1,1,2,2,4"
SCRIPT_FLAGS="--data_dir ../datasets/imagenet/train --data_info_dict_path data/imagenet/to_sample_data_info.pkl \
--batch_size 10 \
--log_dir logs/sr_mean/64_256 --output_dir outputs/sr_var/64_256 \
--model_path checkpoints/sr_mean/64_256/model090000.pt \
--debug_mode yes"

python sr_mean_test.py $MODEL_FLAGS $SCRIPT_FLAGS
```

Testing results will be saved in the directory of `/outputs/sr_var/64_256`.



- Training the variance model

```shell
MODEL_FLAGS="--attention_resolutions 16 --large_size 256 --small_size 64 \
--model_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True \
--use_fp16 True --channel_mult 1,1,2,2,4 --last_layer_type none"
SCRIPT_FLAGS="--data_dir ../datasets/imagenet/train --data_info_dict_path data/sr_imagenet/train_data_info.pkl \
--batch_size 64 --microbatch 8 \
--log_dir logs/sr_var/64_256 --log_interval 10 --save_interval 5000 \
--model_save_dir checkpoints/sr_var/64_256 \
--mean_model_path checkpoints/sr_mean/64_256/model090000.pt --mean_model_args_path logs/sr_mean/64_256mean_model_args.pkl \
--debug_mode yes"

python sr_var_train.py $MODEL_FLAGS $SCRIPT_FLAGS
```

Trained models (checkpoints) will saved in the directory of `checkpoints/sr_var/64_256`.



- Testing the variance model.

```shell
MODEL_FLAGS="--attention_resolutions 16 --large_size 256 --small_size 64 \
--model_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True \
--use_fp16 True --channel_mult 1,1,2,2,4 --last_layer_type none"
SCRIPT_FLAGS="--data_dir ../datasets/imagenet/train --data_info_dict_path data/sr_imagenet/to_sample_data_info.pkl \
--batch_size 10 \
--log_dir logs/sr_var/64_256 --output_dir outputs/sr_var/64_256 \
--model_path checkpoints/sr_var/64_256/model050000.pt \
--mean_model_path checkpoints/sr_mean/64_256/model090000.pt --mean_model_args_path logs/sr_mean/64_256/mean_model_args.pkl \
--debug_mode yes"

python sr_var_test.py $MODEL_FLAGS $SCRIPT_FLAGS
```

Testing results will be saved in the directory of `/outputs/sr_var/64_256`.



- Comparing ddpm-sr model and our trained models.

```shell
python sr_var_verification.py
```

