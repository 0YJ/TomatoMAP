
=======
## TomatoMAP

<p align="center">
  <img src="IMG/index.png" alt="avatar">
</p>

Offical code repository for the manuscript "Tomato Multi-Angle Multi-Pose Dataset for Fine-Grained Phenotyping"

### TomatoMAP Dataset
---------------
You may want to firstly download [TomatoMAP](https://doi.ipk-gatersleben.de/DOI/10bb9f14-ce90-4747-836f-cf61dfb5eea1/e270d2c4-b7fe-4257-ac59-18bf73190adf/2/1416961851) data. We suggest reading [README](https://github.com/0YJ/MPTSTD/blob/main/README.md) before using the dataset.

### TomatoMAP Cascading Processing for Cls, Det
- Clone our repo
```bash
git clone https://github.com/0YJ/EoC.git
cd EoC/code
```

- Put downloaded TomatoMAP into code folder

- Follow the TomatoMAP_builder notebook. Finally you will get TomatoMAP-Cls, TomatoMAP-Det, TomatoMAP-Seg ready for training.
- If you want to train by yourself, you can follow the guide of TomatoMAP_trainer.  

Citation
--------------

Please cite the [TomatoMAP paper](https://arxiv.org/abs/2507.11279) if it helps your research:
```bibtex
@misc{zhang2025tomatomultianglemultiposedataset,
      title={Tomato Multi-Angle Multi-Pose Dataset for Fine-Grained Phenotyping}, 
      author={Yujie Zhang and Sabine Struckmeyer and Andreas Kolb and Sven Reichardt},
      year={2025},
      eprint={2507.11279},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.11279}, 
}
```
