### 代码描述
- 蒸馏过程: [distill.py](./distill.py)
- 蒸馏模块定义: [models/distillation_modules.py](models/distillation_modules.py): 
- 蒸馏参数: [data/hyp.distill.yaml](./data/hyp.distill.yaml)

### 运行方式
1. 安装依赖库
    ``` shell
    pip install -r requirements.txt
    ```

2. 安装[wandb](https://wandb.ai/)
    ``` shell
    pip install wandb
    ```

3. 蒸馏
    ``` shell
    python distill.py
        --teacher <teacher_weights_path>
        --student <student_cfg_path>
        --data <data_description_file_path>
        --workers <num_workers>
        --batch-size <batch_size>
        --epochs-distill <epochs_distill>
        --method <distillation_method>
    ```
    例如
    ``` shell
    python distill.py --teacher yolov5l.pt --student models/yolov5c.yaml --data data/voc.yaml --workers 4 --batch-size 16 --epochs-distill 30 --method AWD
    ```
    注：第一次运行可能需要在[wandb](https://wandb.ai/)注册，并从[https://wandb.ai/authorize](https://wandb.ai/authorize)，获取**API key**并输入
4. 训练
    ``` shell
    python train.py
        --weights <model_weights_path>
        --data <data_description_file_path>
        --workers <num_workers>
        --batch-size <batch_size>
        --epochs <epochs>
    ```
    例如：
    ``` shell
    python train.py --weights runs/distill/exp/weights_distill/last_distill.pt --data data/voc.yaml --workers 4 --batch-size 16 --epochs 100
    ```
### 结果查看
在[wandb](https://wandb.ai/)实时查看蒸馏、训练过程中Loss变化等。
