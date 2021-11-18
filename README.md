# Theorem 1 Verification

## repo 说明

这个代码用于验证定理一的有效性。目前代码的组织形式是为了考虑不同的任务。目前我们使用的任务是「自然图像的超分辨」，因此代码文件（夹）中包含 `sr` 的即是相关的文件。目前具体的代码组织结构如下（按照字母表顺序排列）：

```
theorem-1-verification-python
|----checkpoints                       # 保存所有的模型、优化器
|    |----ddpm_sr                      # 保存已训练好的 ddpm sr 模型
|    |    |----64_256_upsampler.pt     # 64 to 256 ddpm sr model
|    |----sr_mean                      # 保存 sr 均值估计的模型、优化器
|    |    |----64_256                  # 64 to 256 部分
|    |         |----model090000.pt     # 迭代 90000 步的模型
|    |         |----opt090000.pt       # 迭代 90000 步的优化器
|    |----sr_var                       # 保存 sr 方差估计的模型、优化器
|    |    |----64_256                  # 64 to 256 部分
|              |----model050000.pt     # 迭代 50000 步的模型
|              |----opt050000.pt       # 迭代 50000 步的优化器
|----data                              # 保存和训练、测试数据有关的内容
|    |----sr_imagenet                  # sr 方差估计任务所用到的数据集，我们使用 imagenet
|         |----data_info.pkl           # imagenet 数据的所有信息
|         |----to_sample_data_info.pkl # 测试部分数据的信息，一共 2000 图片
|         |----train_data_info.pkl     # 训练部分数据的信息，去掉测试部分后剩下的图片
|----logs                              # 保存日志文件和 tensorboard 输出文件
|    |----ddpm_sr                      # 使用 ddpm sr 模型的采样
|    |    |----64_256_step_250         # 64 to 256, 250 step 采样
|    |----sr_compare                   # 定理一和 ddpm 生成结果的比较
|    |    |----64_256                  # 64 to 256 部分
|    |----sr_mean                      # sr 的均值估计
|    |    |----64_256                  # 64 to 256 部分
|    |         |----mean_model_args.pkl# 均值估计使用的网络的模型参数
|    |----sr_var                       # sr 的方差估计
|         |----64_256                  # 64 to 256 部分
|              |----var_model_args.pkl # 方差估计使用的网络的模型参数
|----models                            # 和模型有关的代码，目前包含的和 sr 任务有关
|    |----__init__.py                  # 可忽略
|    |----nn.py                        # 包含一些网络的小组件
|    |----unet.py                      # 包含 sr 任务中均值估计和方差估计的网络
|----outputs                           # 保存模型使用测试数据时输出的结果
|    |----ddpm_sr                      # ddpm sr 模型的输出结果
|         |----64_256_step_250         # 64 to 256, 250 step 采样
|    |----sr_var                       # sr 任务的输出
|         |----64_256                  # 64 to 256 部分
|----slurms                            # 这个文件夹的内容是在 slurm 系统中提交任务所用的文件，可忽略
|----sr_var                            # sr 方差估计的相关代码（还包含均值估计的部分）
|    |----ddpm                         # 包含和 ddpm sr 有关的代码的文件夹，可以忽略
|    |----__init__.py                  # 可忽略
|    |----base_dataset.py              # 用于 imagenet 图像数据读取的基本数据集
|    |----ddpm_sr_utils.py             # ddpm sr 采样用到的相关函数
|    |----mean_utils.py                # sr 均值估计训练和测试用到的相关函数
|    |----script_util.py               # 包含均值模型和方差模型的基本参数设定
|    |----var_utils.py                 # sr 方差估计训练和测试用到的相关函数
|----utils                             # 包含对于所有任务实验都通用的代码文件
|    |----__init__.py                  # 可忽略
|    |----debug_util.py                # 一些和 debug 有关的函数
|    |----dist_util.py                 # 分布式训练设置的相关函数
|    |----fp16_util.py                 # 混合精度训练设置的相关函数
|    |----logger.py                    # 日志文件和 tensorboard 文件相关的内容
|    |----script_util.py               # 包含一些简化运行代码的函数
|    |----test_util.py                 # 通用的测试模型的过程
|    |----train_util.py                # 通用的训练模型的过程
|----ddpm_sr_sample.py                 # 运行 ddpm sr 采样的代码
|----README.md                         # 说明文档
|----sr_mean_test.py                   # sr 均值估计的测试代码
|----sr_mean_train.py                  # sr 均值估计的训练代码
|----sr_var_test.py                    # sr 方差估计的测试代码
|----sr_var_train.py                   # sr 方差估计的训练代码
|----sr_var_verification.py            # sr 方差估计的验证（和 ddpm sr 的采样结果比较）代码
```

上述的文件树忽略一部分无关紧要的内容。其中，`checkpoints`, `data`, `logs`, `outputs` 是用来保存模型、数据、日志（tensorboard）、输出文件；`models`, `sr_var`, `utils` 是代码相关文件；在根目录下的其他代码文件，基本都是用于运行的代码。下面对这些文件或文件夹进行进一步的说明。

首先，考虑到我们可能会对不同的任务进行实验，但是模型和训练过程在很多地方都是相似的（因为定理一定义的损失函数在不同任务形式上是一致的），所以考虑把各种任务的实验代码都写到一个 repo 中。这样的好处是，每个任务的实验代码，只要处理数据接口的部分，其余不需要重新写，也可以统一修改训练的代码，减少重复工作。因此，在 `checkpoints`, `data`, `logs`, `outputs`  目录下，我们都设一个子文件夹来表示特定的任务。但是这也导致像 `sr_var` 这样的文件夹名在很多地方重复出现，它们都是对应同一个任务的实验，但是保存的文件是不同的，需要注意区分。通常情况，只有一级子目录和二级子目录会出现重复。对于二级子目录，用 `/` 作为前缀，一级子目录，也就是存放代码的文件夹，不使用 `/` 的前缀。

在 `checkpoints`, `logs`, `outputs` 等部分，都有 `64_256` 的路径，这表示该模型是从 64x64 到 256x256 的超分辨实验。我们很可能需要跑其他分辨率的超分辨实验，因此需要进行区分。由于这个任务的核心是估计方差，所以 `outputs` 下仅有 `/sr_var` 子文件夹，均值估计的模型输出也保存在这个文件夹（更多细节在后面的代码使用中有说明）。在 `checkpoints`, `logs` 中，我们则区分了 `/sr_mean` 和 `/sr_var`，因为我们的模型训练分为两步，第一步训练均值模型，第二步训练方差模型。另外，在 `outputs` 文件夹下，还有一个 `sr_compare` 文件夹，这个子文件夹保存的则是我们应用定理一训练的方差模型和 ddpm sr 模型的方差估计进行比较的结果。

`sr_var` 文件夹保存了和 sr 均值估计任务的具体代码。其中 `ddpm` 的参照了原始的代码，原封不动，因此不需要在意其中的内容。



## 代码内容说明

核心抽象部分的代码都在 `utils` 文件夹下。`utils/dist_util.py` 包含了分布式训练的内容，在其他服务器上可能有其他调用的方式，或者每个人喜好的方式不一样，可以对其中的函数进行修改。我是用的是 `torch.distributed.lauch` 的方式。 `utils/fp16_util.py` 是混合精度的训练代码，不需要任何修改，除非是需要在训练过程中保存其他变量到 tensorboard 文件中。关于混合精度的使用，在模型部分，用参数 `use_fp16` 控制了。通常我们为了加快训练和推断，都设 `use_fp16=True`. 这里要注意的是，模型训练和推断时要保持一致的设置。`utils/script_util.py` 包含了几个简单的函数，用于编写 `sr_mean_train.py` 之类的运行程序。

 `utils` 文件夹下，`train_util.py` 和 `test_util.py` 是最重要的。这两个代码文件分别包含了 `TrainLoop` 和 `TestLoop` 两个抽象父类，封装了训练和测试的过程（我觉得对于不同的任务，这两个过程大体是一样的，故而把它们抽象出来）。其中，允许自定义的 `_post_process` 内部方法，可以为不同的任务编写特定的处理代码。这种处理相对简单，灵活性一般，不过处理大多数任务应该是足够的。

接下来对超分辨实验的 `sr_var` 部分的代码进行一些说明。由于我们需要使用 ddpm 的 sr 模型来生成 ground truth，所以包含了 `/ddpm` 子文件夹。这个文件夹包含了相关的代码，`sr_var/ddpm_sr_utils.py` 是为了生成方便所写的一个特殊的数据集接口。主目录下的 `ddpm_sr_sample.py` 则是对应的运行代码文件。`sr_var/base_dataset.py` 给出了一个 imagenet 数据集的通用的基本接口（可以进一步修改），用于均值模型和方差模型的训练和测试。`sr_var/mean_utils.py` 和 `sr_var/var_utils.py` 则利用 `TrainLoop` 和 `TestLoop` 写了在超分辨任务实验中的训练和测试子类 `SRMeanTrainLoop` 等，还包括了更具体的数据接口。`sr_var/script_util.py` 包含了均值模型和方差模型的基本参数设置。

主目录下的运行代码文件的说明，请参考下一节的部分。



## `sr_var` 部分的代码使用说明

这个部分，我们会对超分辨任务 64 到 256 分辨率的实验进行说明。对于 128 到 512 的实验，将会在每一个步骤的说明中进行补充，指明哪些部分不需要修改，哪些部分修改为 128 to 512 实验相关的内容，并给出一个基本样例。

这个部分也会对数据、模型保存、模型输出保存进行更多、更详细的说明。

在运行代码的说明部分，省略和分布式相关的内容。

### 1 数据准备

从 https://image-net.org 下载 ImageNet 数据，应该可以用这个链接 https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar，这是原本 ILSVRC 的训练数据，我们仅仅使用这个部分就可以了。下载后是一个 tar 压缩文件。解压之后，将会包含 1000 个 tar 文件，每个文件对应一个类别的图像数据。使用下面的 shell 脚本解压并删除压缩包。

```shell
for file in *.tar; do tar -xf "$file"; rm "$file"; done
```

解压完成之后，每个图片文件的文件名类似于 `n01440764_xxxxxx.JPEG`，`n01440764` 表示类别。

这个数据集所在的路径在后续的代码运行中会用到，因为这个数据集太大了，不太适合事先处理整个数据集。我采用的方式是读取数据的时候再进行处理（比如裁剪之类的）。

我保存的路径是 `../datasets/imagenet/train` （未保存在 repo 的主目录下）, 这个根据实际情况修改。



### 2 Package Requirements

除了常见的 python 库，可能需要额外安装如下的库：

```
blobfile  # 读取文件
tensorboardX  # tensorboard 文件输出
socket  # 用来自动找端口的，在 utils/dist_util.py 里面用到，去掉也可以。
pynvml  # 用来看 GPU 使用情况，去掉也可以
```

pytorch 的版本应该是 1.6 到 1.9 都可以跑通，但是有的版本可能会有 warnings.



### 3 ddpm sr 生成

ddpm sr model 可以从这里下载：https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64_256_upsampler.pt (64 to 256)，https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128_512_upsampler.pt (128 to 512).

由于参数较多，故对其划分，方便调整（其他亦同）。

运行 **64 to 256** 的程序脚本如下（ddpm sr sampling for 64 to 256）：

```shell
cd /home/ytxie/theorem-1-verification-python

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

`GD_FLAGS` 不用改动，`MODEL_FLAGS`  需要修改的是 `--large_size` 和 `--small_size`，网络参数可能也要调整一下。

`SCRIPT_FLAGS` 部分需要改动较多。

1. `--log_dir`, `output_dir`, `--model_path_checkpoints` 需要根据实际情况改动。
2. `--batch_size` 取决于 GPU 显存大小。

每个测试图像跑 100 个 sample 非常花时间，不过可以随时终止程序。重新运行程序会自动继续生成。

下面给出 **128 to 512** 的程序脚本**样例**（ddpm sr sampling for 128 to 512 example）：

```shell
cd /home/ytxie/theorem-1-verification-python

MODEL_FLAGS="--attention_resolutions 32,16 --class_cond True --large_size 512 --small_size 128 \
--num_channels 192 --num_head_channels 64 --num_res_blocks 2 --learn_sigma True --resblock_updown True \
--use_fp16 True --use_scale_shift_norm True"
GD_FLAGS="--diffusion_steps 1000 --noise_schedule linear --learn_sigma True --timestep_respacing 250"
SCRIPT_FLAGS="--log_dir logs/ddpm_sr/128_512_step_250 --output_dir outputs/ddpm_sr/128_512_step_250 \
--data_dir ../datasets/imagenet/train --model_path checkpoints/ddpm_sr/128_512_upsampler.pt \
--to_sample_images_dict_path data/sr_imagenet/to_sample_data_info.pkl \
--num_samples_per_image 100 --batch_size 5"

python ddpm_sr_sample.py $MODEL_FLAGS $GD_FLAGS $SCRIPT_FLAGS
```



### 均值模型的训练和测试

运行 **64 to 256** 的训练程序脚本如下（mean model train for 64 to 256）：

```shell
cd /home/ytxie/theorem-1-verification-python

MODEL_FLAGS="--attention_resolutions 16 --large_size 256 --small_size 64 \
--model_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True \
--use_fp16 True --channel_mult 1,1,2,2,4"
SCRIPT_FLAGS="--data_dir ../datasets/imagenet/train --data_info_dict_path data/sr_imagenet/train_data_info.pkl \
--batch_size 64 --microbatch 8 \
--run_time 47.8 \
--log_dir logs/sr_mean/64_256 --log_interval 10 --save_interval 5000 \
--model_save_dir checkpoints/sr_mean/64_256 --resume_checkpoint model090000.pt \
--debug_mode yes"

python sr_mean_train.py $MODEL_FLAGS $SCRIPT_FLAGS
```

`MODEL_FLAGS` 的参数可以自由设定，不过目前的设定看起来效果还可以。需要修改的就是 `--large_size` 和 `--small_size`. 当分辨率提高时，可能会考虑增加网络的深度，这个由 `--channel_mult` 参数控制。这里的 `1,1,2,2,4` 数字表示不同深度的 channel 的维数。

`SCRIPT_FLAGS` 需要修改的较多。

 `--log_dir`, `--model_save_dir` 根据实际情况调整。

对于初次训练，`--rusume_checkpoint` 参数可以忽略，如果是基于此前训练的模型继续训练，则需要使用这个参数。

`--microbatch` 是用于当 GPU 显存较小时，用更小的批量，多次计算，用累计梯度来更新网络参数。如果不使用 `--microbatch` 忽略或者设为 `--microbatch -1` 即可。

`--run_time` 参数是为了在北大的服务器上运行时方便保存模型特别设置的，可以忽略（默认为 -1，对实验无影响）。如果要控制训练的最大步数，可以使用 `--max_step` 参数，它的默认值非常大（相当于一直训练）。

`--debug_mode` 表示输出 GPU 的使用情况，也可以忽略（默认是 False/no）。

根据实际情况和个人喜好可以调整的是 `--batch_size`, `--log_interval`, `--save_interval`. `--log_interval` 和 tensorboard 监测 loss 的频率有关（每隔多少步迭代记录一次），`--save_interval` 表示保存模型的频率。保存模型的同时会保存一个优化器状态，方便重新从这个模型开始训练。

模型和优化器状态会保存在 `--model_save_dir` 对应的文件夹下。日志文件则保存在 `--log_dir` 文件夹下。初次训练时，会在 `--log_dir` 文件夹下保存一个 `mean_model_args.pkl` 的文件，保存当前训练模型的参数设定（从某个 resume checkpoint 开始训练时，会自动加载 `mean_model_args.pkl` 文件获取模型参数，此时 `MODEL_FLAGS` 的设定失效）。

运行 **64 to 256** 的测试程序脚本如下（mean model test for 64 to 256）：

```shell
cd /home/ytxie/theorem-1-verification-python

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

在测试的时候，我们会输出测试集图像的超分辨均值估计的结果。输出结果保存在 `--output_dir` 下，每个图片对应一个子文件夹，保存有均值估计和模型输入（低分辨率图像）两个图片结果。在日志文件的 tensorboard 中也会保留这些结果，方便快速浏览。

实际上 `MODEL_FLAGS` 没有什么用（只是因为复制粘贴所以保留了），因为模型的参数由 `--log_dir` 文件夹下的 `mean_model_args.pkl` 保存着（训练的时候会产生），我们会直接调用这个参数来创建模型。

`SCRIPT_FLAGS` 需要注意一些参数。

`--log_dir` 必须和训练时的参数保持一致。

`--output_dir` 可能需要修改。

`--model_path` 是训练时模型保存的路径，需要和训练的结果保持一致。

`--debug_mode` 可以忽略。

下面给出 **128 to 512** 均值模型的训练、测试程序脚本**样例**（mean model train and test for 128 to 512 **example**）：

不改变模型参数的设置。

训练：

```shell
cd /home/ytxie/theorem-1-verification-python

MODEL_FLAGS="--attention_resolutions 16 --large_size 512 --small_size 128 \
--model_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True \
--use_fp16 True --channel_mult 1,1,2,2,4"
SCRIPT_FLAGS="--data_dir ../datasets/imagenet/train --data_info_dict_path data/sr_imagenet/train_data_info.pkl \
--batch_size 64 --microbatch -1 \
--max_step 100000 \
--log_dir logs/sr_mean/128_512 --log_interval 10 --save_interval 5000 \
--model_save_dir checkpoints/sr_mean/128_512 \
--debug_mode yes"

python sr_mean_train.py $MODEL_FLAGS $SCRIPT_FLAGS
```

测试

```shell
cd /home/ytxie/theorem-1-verification-python

MODEL_FLAGS="--attention_resolutions 16 --large_size 512 --small_size 128 \
--model_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True \
--use_fp16 True --channel_mult 1,1,2,2,4"
SCRIPT_FLAGS="--data_dir ../datasets/imagenet/train --data_info_dict_path data/imagenet/to_sample_data_info.pkl \
--batch_size 10 \
--log_dir logs/sr_mean/128_512 --output_dir outputs/sr_var/128_512 \
--model_path checkpoints/sr_mean/128_512/model090000.pt \
--debug_mode yes"

python sr_mean_test.py $MODEL_FLAGS $SCRIPT_FLAGS
```



### 方差模型的训练和测试

方差模型和均值模型在参数上几乎是一样，只是方差模型的训练和测试的时候，需要给定一个均值模型，因此增加了额外的一些参数。这里仅仅对增加的参数进行说明。

**64 to 256** 方差模型的训练和测试程序脚本如下（mean model train and test for 64 to 256）：

训练：

```shell
cd /home/ytxie/theorem-1-verification-python

MODEL_FLAGS="--attention_resolutions 16 --large_size 256 --small_size 64 \
--model_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True \
--use_fp16 True --channel_mult 1,1,2,2,4 --last_layer_type none"
SCRIPT_FLAGS="--data_dir ../datasets/imagenet/train --data_info_dict_path data/sr_imagenet/train_data_info.pkl \
--batch_size 64 --microbatch 8 \
--run_time 47.8 \
--log_dir logs/sr_var/64_256 --log_interval 10 --save_interval 5000 \
--model_save_dir checkpoints/sr_var/64_256 --resume_checkpoint model050000.pt \
--mean_model_path checkpoints/sr_mean/64_256/model090000.pt --mean_model_args_path logs/sr_mean/64_256mean_model_args.pkl \
--debug_mode yes"

python sr_var_train.py $MODEL_FLAGS $SCRIPT_FLAGS
```

测试：

```shell
cd /home/ytxie/theorem-1-verification-python

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

首先是 `MODEL_FLAGS` 多了一个参数 `--last_layer_type`，这是用于在网络的输出部分增加一个函数来限制方差估计的值域。目前采用的是 "none", 也就是什么也不加，这可能会导致推断的时候预测的某些像素的方差是负数。这种情况，我们可以用后处理将其设为 0。不过增加一个限制似乎更好，因为方差本身必须大于等于 0，而且也是有上限的。可以选择的另外的参数是 "sigmoid" 和 "exp"，可以试试这两个选择的效果。

`SCRIPT_FLAGS` 的部分，增加了 `--mean_model_path` 和 `--mean_model_args_path` 两个参数。它们分别是均值模型的路径和均值模型的参数文件的路径（这样就不需要额外为均值模型提供参数了）。

测试跑完之后会在 `--output_dir` 文件夹下每个图片对应的子文件中增加一个方差估计的 pkl 文件，保存方差估计的值。 `--output_dir` 文件夹和均值模型测试时的文件夹要保持一致。

下面给出 **128 to 512** 方差模型的训练、测试程序脚本**样例**（var model train and test for 128 to 512 **example**）：

训练：

```shell
cd /home/ytxie/theorem-1-verification-python

MODEL_FLAGS="--attention_resolutions 16 --large_size 512 --small_size 128 \
--model_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True \
--use_fp16 True --channel_mult 1,1,2,2,4 --last_layer_type none"
SCRIPT_FLAGS="--data_dir ../datasets/imagenet/train --data_info_dict_path data/sr_imagenet/train_data_info.pkl \
--batch_size 64 --microbatch -1 \
--max_step 100000 \
--log_dir logs/sr_var/128_512 --log_interval 10 --save_interval 5000 \
--model_save_dir checkpoints/sr_var/128_512 \
--mean_model_path checkpoints/sr_mean/128_512/model090000.pt --mean_model_args_path logs/sr_mean/128_512/mean_model_args.pkl \
--debug_mode yes"

python sr_var_train.py $MODEL_FLAGS $SCRIPT_FLAGS
```

测试：

```shell
cd /home/ytxie/theorem-1-verification-python

MODEL_FLAGS="--attention_resolutions 16 --large_size 512 --small_size 128 \
--model_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True \
--use_fp16 True --channel_mult 1,1,2,2,4 --last_layer_type none"
SCRIPT_FLAGS="--data_dir ../datasets/imagenet/train --data_info_dict_path data/sr_imagenet/to_sample_data_info.pkl \
--batch_size 10 \
--log_dir logs/sr_var/128_512 --output_dir outputs/sr_var/128_512 \
--model_path checkpoints/sr_var/128_512/model050000.pt \
--mean_model_path checkpoints/sr_mean/128_512/model090000.pt --mean_model_args_path logs/sr_mean/128_512/mean_model_args.pkl \
--debug_mode yes"

python sr_var_test.py $MODEL_FLAGS $SCRIPT_FLAGS
```



### 和 ddpm sr 模型的比较和验证

**64 to 256** 的验证运行程序脚本如下：

```shell
cd /home/ytxie/theorem-1-verification-python
python sr_var_verification.py --data_dir ../datasets/imagenet/train --data_info_dict_path data/sr_imagenet/to_sample_data_info.pkl --large_size 256 --ddpm_sr_output_dir outputs/ddpm_sr/64_256_step_250 --num_samples 100 --mean_model_step 90000 --var_model_step 50000 --log_dir logs/sr_compare/64_256
```

这个程序不需要使用 GPU.

大部分的参数和之前的参数设定、训练的结果有关。

`ddpm_sr_output_dir` 是保存 ddpm sr 模型生成结果的文件夹。`num_samples` 表示 ddpm sr 模型每个图片生成的样本数量，我们和此前保持一致，都是 100. `--mean_model_step` 和  `--var_model_step` 分别指均值模型估计的结果、方差模型估计的结果是由迭代了多少步的模型产生的（我们用这个数字区分不同模型的输出，比如 `model090000.pt` 的输出结果文件名会带有 `90000`）。这两个参数和实际的模型选择有关。日志文件和 tensorboard 文件保存在 `--log_dir` 指定的文件夹中。日志文件会输出均值模型估计的准确性和方差模型估计的准确性（和 ddpm sr 的 100 个样本估计的结果进行比较）。tensorboard 文件则会包含所有测试图片的均值估计、方差估计等所有的结果。

**128 to 512** 的验证运行程序脚本**样例**如下：

```shell
cd /home/ytxie/theorem-1-verification-python
python sr_var_verification.py --data_dir ../datasets/imagenet/train --data_info_dict_path data/sr_imagenet/to_sample_data_info.pkl --large_size 512 --ddpm_sr_output_dir outputs/ddpm_sr/128_512_step_250 --num_samples 100 --mean_model_step 90000 --var_model_step 50000 --log_dir logs/sr_compare/128_512
```



## 编写其他任务实验的代码的注意事项

实际上完全可以仿照 `sr_var` 文件夹下的代码结构来编写。主要分为几个内容：

1. 数据集的处理。数据集可以保存在 `data` 文件夹下，也可以保存在服务器的其他区域，但必须确定访问路径。我们只要编写一个数据接口（类似于 `sr_var/base_dataset.py` ）的代码即可。
2. 编写一个关于该任务的训练和测试的子类。父类 `TrainLoop` 和 `TestLoop` 已经包含了训练和测试的基本过程，需要补充的仅仅是对于输出、debug 等额外自定义的控制的内容。这个部分需要在子类中自定义 `_post_process` 内部方法。
3. 如果需要编写额外的模型，可以保存在 `models` 文件夹下。也可以直接利用现有的模型。
4. `/script_util.py` 代码通常保存的是模型的基本参数设置，避免在训练和测试的主程序中占用太多空间。通常模型的大部分参数是固定不变的，因此单独写一个文件保存比较方便。
5. 在主目录下编写训练和测试的代码，基本结构和 `sr_mean_test.py`, `sr_mean_train.py` 类似。使用的时候调用 `TrainLoop` 或者 `TestLoop` 的 `run_loop()` 方法就可以了。

当然，也可以按照自己的喜好来编写，只不过 `TrainLoop` 和 `TestLoop` 可以直接利用起来，这样会方便非常多。

