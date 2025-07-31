# Inter-Rater Agreement Analysis Tool

This tool calculates and visualizes **Fleiss' Kappa** and **Cohen's Kappa** scores for inter-rater agreement based on object detection bounding box annotations.

It supports:
- Penalized and unpenalized Fleiss' Kappa
- Pairwise Cohen's Kappa with optional penalization
- Agreement heatmap visualization of annotation density
- Summary matrix and plots for comparative analysis

---

## Requirements

Install the required Python libraries before use:

```bash
pip install -r requirements.txt
```
## Directory Structure
The input label files should follow yolo format, with:
**class_id, x_center, y_center, width, height**

```bash
data/
├── Annotator1/
│   ├── image1.txt
│   └── image2.txt
├── Annotator2/
│   ├── image1.txt
│   └── image2.txt
...
```

## Usage
```bash
python avh.py -input data -out output [-size WIDTH HEIGHT] [-iou IOU_THRESHOLD] [-p (optional)]
```
example: 
```bash
python avh.py -input data -out output -size 1920 1080 -iou 0.1 -p
```
