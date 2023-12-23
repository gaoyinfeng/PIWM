# Dream to Drive with Predictive Individual World Model
JAX implementation of the **PIWM**. PIWM is a novel model-based reinforcement learning method that is typically designed for autonomous driving tasks. It learns to **model the driving environment from an individual perspective** and enhances the transition dynamic by explicitly **modeling the interactive relations between vehicles**. Trajectory prediction further replaces observation reconstruction for representation learning to **better capture the future intentions** or motion trends of interested vehicles within the latent states.

[**Project page**][website_project]

The simulation environment used in this work is [**I-SIM simulator**][website_ISIM], which is built upon [Interaction Dataset][website_INTER].

We plan to release the codes after the paper is accepted!

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



## Acknowledgement
We appreciate the following GitHub repos for their valuable code base or dataset:

https://github.com/danijar/dreamerv3

https://github.com/fzi-forschungszentrum-informatik/Lanelet2

https://github.com/interaction-dataset/interaction-dataset
