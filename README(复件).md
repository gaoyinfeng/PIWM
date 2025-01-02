# Dream to Drive with Predictive Individual World Model
JAX implementation of the **PIWM**. PIWM is a novel model-based reinforcement learning method built upon DreamerV3. It is typically designed for autonomous driving tasks. It learns to **model the driving environment from an individual perspective** and enhances the transition dynamic by explicitly **modeling the interactive relations between vehicles**. Trajectory prediction further replaces observation reconstruction for representation learning to **better capture the future intentions** or motion trends of interested vehicles within the latent states.

[**Project page**][website_project]

The simulation environment used in this work is [**I-SIM simulator**][website_ISIM], which is built upon [Interaction Dataset][website_INTER].

The paper has been accepted! We plan to release the codes in a month!

[website_project]: https://sites.google.com/view/piwm
[website_ISIM]: https://github.com/gaoyinfeng/I-SIM/
[website_INTER]: http://www.interaction-dataset.com/

- Overall Framework and Differences from Original Dreamer:
<div align=center>
<img width="85%" src="https://github.com/gaoyinfeng/PIWM/blob/main/pics/overall.png">
</div>

- Detailed Structure of Models:
<div align=center>  
<img width="85%" src="https://github.com/gaoyinfeng/PIWM/blob/main/pics/detailed.png">
</div>


## Instructions

The code has been tested on Linux and requires Python 3.11+.

### Docker

You can either use the provided `Dockerfile` that contains instructions or follow the manual instructions below.

### Manual

Install [JAX](https://github.com/google/jax#pip-installation-gpu-cuda) and then the other dependencies:

```shell
pip install -U -r requirements.txt
```

Training script:

```shell
python train.py --logdir {logdir} --configs interaction_prediction --task interaction_prediction
```

通过修改`dreamerv3/configs.yaml`中的`"script"`可以选择进行训练或者测试:

```yaml
script: eval_only
```

Test scripts:

```shell
python train.py --logdir {logdir} --configs interaction_prediction --task interaction_prediction
```

同时需要指定`checkpoint`文件:

```yaml
from_checkpoint: {checkpoint_dir}/{checkpoint}.ckpt
```

此外，通过修改`dreamerv3/configs.yaml`中的`"loader_type"`以选择大规模数据集或小规模数据集的训练或测试:

Large_scale:

```yaml
loader_type: large_scale
```

Small_scale:

```yaml
loader_type: small_scale
```



## Catalog

- [ ] Code & Checkpoints Release
- [ ] Initialization

## Acknowledgement
We appreciate the following GitHub repos for their valuable code base or dataset:

https://github.com/danijar/dreamerv3

https://github.com/fzi-forschungszentrum-informatik/Lanelet2

https://github.com/interaction-dataset/interaction-dataset
