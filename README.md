# 🍅 TomatoMAP: Tomato Multi-Angle Multi-Pose Dataset for Fine-Grained Phenotyping

<p align="center">
<a href="https://scholar.google.com/">Yujie Zhang</a>,
<a href="https://scholar.google.com/">Sabine Struckmeyer</a>,
<a href="https://scholar.google.com/">Andreas Kolb</a>,
<a href="https://scholar.google.com/">Sven Reichardt</a>
</p>

<p align="center">
<a href="https://0yj.github.io/tomato_map/">🌐[Project Page]</a>
<a href="https://arxiv.org/abs/2507.11279">📄[Paper]</a>
<a href="https://github.com/0YJ/TomatoMAP">💻[Code]</a>
<a href="https://doi.ipk-gatersleben.de/DOI/89386758-8bfd-41ca-aa9c-ee363e9d94c9/073051f0-b05e-4b43-a9cd-0435fe7cd913/2/1847940088">📁[Dataset]</a>
</p>

<p style="align:justify"><b>Abstract</b>: Observer bias and inconsistencies in traditional plant phenotyping methods limit the accuracy and reproducibility of fine-grained plant analysis. To overcome these challenges, we developed TomatoMAP, a comprehensive dataset for Solanum lycopersicum using an Internet of Things (IoT) based imaging system with standardized data acquisition protocols. Our dataset contains 64,464 RGB-images that capture 12 different plant poses from four camera elevation angles. Each image includes manually annotated bounding boxes for seven regions of interest (ROIs), including leaves, panicle, batch of flowers, batch of fruits, axillary shoot, shoot and whole plant area, along with 50 fine-grained growth stage classifications based on the BBCH scale. Additionally, we provide 3,616 high-resolution image subset with pixel-wise semantic and instance segmentation annotations. We validated our dataset using a cascading model deep learning framework combining different models. Through AI vs. Human analysis involving five domain experts, we demonstrate that the models trained on our dataset achieve accuracy comparable to the experts. Cohen's Kappa and inter-rater agreement heatmap confirm the reliability of automated fine-grained phenotyping using our approach.</p>

## 📢 Updates

* 15.07.2025: Paper available on [arXiv](https://arxiv.org/abs/2507.11279)
* 18.07.2025 Full dataset release on e!DAL
* Coming soon: Nature Scientific Data

## Getting Started
Please check code subfolder for more details.

## Citation

If you use TomatoMAP in your research, please cite:

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


@dataset{tomatomap,
  title={TomatoMAP: Tomato Multi-Angle Multi-Pose Dataset for Fine-Grained Phenotyping},
  author={Yujie Zhang, Sabine Struckmeyer, Andreas Kolb, and Sven Reichardt},
  journal={e!DAL - Plant Genomics and Phenomics Research Data Repository (PGP)},
  year={2025}
}
```

## License

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Commercial use requires permission.
