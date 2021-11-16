#!/bin/bash
#SBATCH -J ddpm_sr          # Job名
#SBATCH -o ./sysout.txt  # 输出, 目录./out必须存在, 否则无法成功提交job. 也可删除此行由系统自动指定.
#SBATCH --qos=short       # qos(quality of service): normal (1 job, 1 gpu), short (3 job, 3 gpu), debug (1 job, 1 gpu)
#SBATCH -p RTX3090        # 指定partition: V100(gpu00), RTX3090(gpu01), geforce(gpu02, gpu03), etc.
#SBATCH --nodelist=gpu01   # 指定属于上述partition的特定节点. 也可删除这一行, 由系统自动分配.
#SBATCH --cpus-per-task=4  # 申请 cpu core 数; 可用内存与申请 cpu core 数成正比.
#SBATCH --mem=20G
#SBATCH --gres=gpu:3      # 申请 gpu 数
#SBATCH -N 1               # 申请节点数,一般为1
#SBATCH -t 2-00:00:00       # 申请Job运行时长0小时5分钟0秒, 若要申请超过一天时间, 如申请1天, 书写格式为#SBATCH -t 1-00:00:00
# 上述 SBATCH 参数不指定时均有系统指定的默认值

# 随着 Job 的提交和执行, slurm 会帮助用户在申请的节点上挨个执行下述命令

module add anaconda/3
source activate pytorch1.9
python ../get_device_configuration_info.py --slurm_file run.slurm
file=../slurm_param.txt
line=($(awk 'NR==1 {print $0}' $file))
nproc_per_node=${line:15}
line=($(awk 'NR==2 {print $0}' $file))
master_addr=${line:12}
line=($(awk 'NR==3 {print $0}' $file))
master_port=${line:12}

cd Home/ytxie/theorem-1-verification

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --large_size 256 --small_size 64 \
--num_channels 192 --num_heads 4 --num_res_blocks 2 --learn_sigma True --resblock_updown True \
--use_fp16 False --use_scale_shift_norm True"
GD_FLAGS="--diffusion_steps 1000 --noise_schedule linear --learn_sigma True --timestep_respacing 250"
SCRIPT_FLAGS="--log_dir logs/ddpm_sr/step_250 --output_dir outputs/ddpm_sr/step_250 \
--data_dir ../datasets/imagenet/train --model_path checkpoints/ddpm_sr/64_256_upsampler.pt \
--to_sample_images_dict_path data/to_sample_data_info.pkl \
--num_samples_per_image 1 --batch_size 1"

python -m torch.distributed.launch \
--nnodes=1 \
--node_rank=0 \
--nproc_per_node=$nproc_per_node \
--master_addr=$master_addr \
--master_port=$master_port \
ddpm_sr_sample.py $MODEL_FLAGS $GD_FLAGS $SCRIPT_FLAGS
