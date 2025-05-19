
import os
import sys

folder = "/home/gaverta/VRP-SAM"
if not os.path.abspath(os.curdir) == folder: sys.exit()

NGPUS = 1
BS = 16
grid = [    
    {'name': f'vrpsam2_coco_fold0', ' --benchmark': 'coco'},    
    {'name': f'vrpsam2_coco_fold1', ' --benchmark': 'coco'},    
    {'name': f'vrpsam2_coco_fold2', ' --benchmark': 'coco'},    
    {'name': f'vrpsam2_coco_fold3', ' --benchmark': 'coco'},    

    {'name': f'vrpsam2_lvis_fold0', ' --benchmark': 'lvis'},    
    {'name': f'vrpsam2_lvis_fold1', ' --benchmark': 'lvis'},    
    {'name': f'vrpsam2_lvis_fold2', ' --benchmark': 'lvis'},    
    {'name': f'vrpsam2_lvis_fold3', ' --benchmark': 'lvis'},    
    {'name': f'vrpsam2_lvis_fold4', ' --benchmark': 'lvis'},    
    {'name': f'vrpsam2_lvis_fold5', ' --benchmark': 'lvis'},    
    {'name': f'vrpsam2_lvis_fold6', ' --benchmark': 'lvis'},    
    {'name': f'vrpsam2_lvis_fold7', ' --benchmark': 'lvis'},    
    {'name': f'vrpsam2_lvis_fold8', ' --benchmark': 'lvis'},    
    {'name': f'vrpsam2_lvis_fold9', ' --benchmark': 'lvis'},    
]

for arg_set in grid:
    add_args = ''
    exp_name = arg_set['name']
    for arg_s, arg_v in arg_set.items():
        if arg_s != 'name':
            add_args += f'{arg_s} {arg_v}'
        
    fold = int(exp_name.split('_fold')[-1].split('_')[0])
    add_args += f' --fold {fold}'
    if 'box' in exp_name:
        add_args += ' --condition box'
    if 'point' in exp_name:
        add_args += ' --condition point'
    if 'scribble' in exp_name:
        add_args += ' --condition scribble'
    if 'cmulti' in exp_name:
        add_args += ' --condition multi'

    CONTENT = \
f"""#!/bin/bash 
#SBATCH --job-name=EXP_NAME
#SBATCH --gres=gpu:{NGPUS}
#SBATCH -N 1
#SBATCH --ntasks-per-node={NGPUS*8}
#SBATCH --partition=fair_gpu
#SBATCH --mem=50GB
#SBATCH --time=24:00:00
#SBATCH --output {folder}/out_files/out_EXP_NAME.txt
#SBATCH --error {folder}/out_files/err_EXP_NAME.txt

module load miniforge/24.3.0-0
conda activate /home/gaverta/sam2_env
export WANDB_API_KEY="138defcbedc69ea57f4884949f29ce11311edb44"

cd {folder}
port=$(python get_free_port.py)
python  -m torch.distributed.launch --nproc_per_node={NGPUS} --master_port=${{port}} --use_env train.py \
    --datapath /leonardo_scratch/fast/IscrB_LarGEO/fast_dataset  \
    --logpath EXP_NAME --backbone resnet50 --condition mask --num_query 50 \
    --epochs 50 --lr 1e-4  --bsz {BS} {add_args}
"""

    filename = f"{folder}/jobs/{exp_name}.sh"
    content = CONTENT.replace("EXP_NAME", exp_name)
    with open(filename, "w") as file:
        _ = file.write(content)
    _ = os.system(f"sbatch {filename}")
    print(f"sbatch {filename}")


