#!/bin/bash
#SBATCH -J theorem_1_mnist_classification_train          # Job名
#SBATCH -o ./sysout.txt  # 输出, 目录./out必须存在, 否则无法成功提交job. 也可删除此行由系统自动指定.
#SBATCH --qos=short       # qos(quality of service): normal (1 job, 1 gpu), short (3 job, 3 gpu), debug (1 job, 1 gpu)
#SBATCH -p geforce        # 指定partition: V100(gpu00), RTX3090(gpu01), geforce(gpu02, gpu03), etc.
#SBATCH --nodelist=gpu03   # 指定属于上述partition的特定节点. 也可删除这一行, 由系统自动分配.
#SBATCH --cpus-per-task=4  # 申请 cpu core 数; 可用内存与申请 cpu core 数成正比.
#SBATCH --mem=4G
#SBATCH --gres=gpu:1      # 申请 gpu 数
#SBATCH -N 1               # 申请节点数,一般为1
#SBATCH -t 1-00:00:00       # 申请Job运行时长0小时5分钟0秒, 若要申请超过一天时间, 如申请1天, 书写格式为#SBATCH -t 1-00:00:00
# 上述 SBATCH 参数不指定时均有系统指定的默认值

# 随着 Job 的提交和执行, slurm 会帮助用户在申请的节点上挨个执行下述命令

module add anaconda/3
source activate pytorch1.9

cd /home/ytxie/theorem-1-verification-python

SCRIPT_FLAGS="--model_save_dir checkpoints/mnist_classification/model_noise_label_0.3 \
--task train_on_noisy_label --alpha 0.3 \
--noise_type model --noise_label_file model_noisy_label_0.3.pkl"

python mnist_classification.py $SCRIPT_FLAGS
