import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import rcParams
from statsmodels.stats.inter_rater import fleiss_kappa
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
import pandas as pd

# font settings for publishing
rcParams['font.family'] = 'Calibri'
rcParams['font.size'] = 8

# read labels
def read_label(file_path):
    bboxes = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            class_id = int(data[0])
            x, y, w, h = map(float, data[1:])
            bboxes.append((class_id, x, y, w, h))
    return bboxes

# iou caculation based on label info
def compute_iou(box1, box2):
    _, x1, y1, w1, h1 = box1
    _, x2, y2, w2, h2 = box2
    xa1, ya1, xa2, ya2 = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
    xb1, yb1, xb2, yb2 = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area, box2_area = (xa2 - xa1)*(ya2 - ya1), (xb2 - xb1)*(yb2 - yb1)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0

# greedy iou algorithm
def greedy_iou(boxes_a, boxes_b, iou_threshold=0.5):
    matched_indices = []
    unmatched_a = set(range(len(boxes_a)))
    unmatched_b = set(range(len(boxes_b)))

    iou_matrix = np.zeros((len(boxes_a), len(boxes_b)))
    for i, box_a in enumerate(boxes_a):
        for j, box_b in enumerate(boxes_b):
            iou_matrix[i, j] = compute_iou(box_a, box_b)

    while True:
        max_iou = np.max(iou_matrix)
        if max_iou < iou_threshold:
            break

        idx_a, idx_b = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        matched_indices.append((idx_a, idx_b))
        unmatched_a.remove(idx_a)
        unmatched_b.remove(idx_b)

        iou_matrix[idx_a, :] = -1
        iou_matrix[:, idx_b] = -1

    return matched_indices, list(unmatched_a), list(unmatched_b)

# fleiss kappa
def fleiss_kappa(table):
    try:
        unique_rows = np.unique(table, axis=0)
        if unique_rows.shape[0] == 1:
            if unique_rows[0, 0] == 0:
                return 1.0
            else:
                return 0.0
        kappa = fleiss_kappa(table)
        if np.isnan(kappa):
            return 0.0
        return kappa
    except ZeroDivisionError:
        return 0.0

# fleiss table
def penalized_fleiss_table(all_boxes, iou_threshold=0.5, num_classes=7):
    num_annotators = len(all_boxes)
    all_matched_centers = []

    for i in range(num_annotators):
        for j in range(i+1, num_annotators):
            matches, _, _ = greedy_iou(all_boxes[i], all_boxes[j], iou_threshold)
            for idx1, idx2 in matches:
                box_i = all_boxes[i][idx1]
                box_j = all_boxes[j][idx2]
                center = ((box_i[1] + box_j[1]) / 2, (box_i[2] + box_j[2]) / 2)
                all_matched_centers.append(center)

    unique_targets = []
    for center in all_matched_centers:
        if not any(np.linalg.norm(np.array(center) - np.array(u)) < 0.01 for u in unique_targets):
            unique_targets.append(center)

    table = []
    for center in unique_targets:
        votes = [0] * (num_classes + 1)  # +1 unlabeled
        for boxes in all_boxes:
            matched_label = None
            for box in boxes:
                dist = np.linalg.norm(np.array([box[1], box[2]]) - np.array(center))
                if dist < 0.01:
                    matched_label = box[0]
                    break
            if matched_label is not None:
                votes[matched_label] += 1
            else:
                votes[-1] += 1  # unlabeled class vote

        table.append(votes)
    return np.array(table)

def unpenalized_fleiss_table(all_boxes, iou_threshold=0.5, num_classes=7):
    num_annotators = len(all_boxes)
    all_matched_centers = []

    for i in range(num_annotators):
        for j in range(i+1, num_annotators):
            matches, _, _ = greedy_iou(all_boxes[i], all_boxes[j], iou_threshold)
            for idx1, idx2 in matches:
                box_i = all_boxes[i][idx1]
                box_j = all_boxes[j][idx2]
                center = ((box_i[1] + box_j[1]) / 2, (box_i[2] + box_j[2]) / 2)
                all_matched_centers.append(center)

    unique_targets = []
    for center in all_matched_centers:
        if not any(np.linalg.norm(np.array(center) - np.array(u)) < 0.01 for u in unique_targets):
            unique_targets.append(center)

    table = []
    for center in unique_targets:
        votes = [0] * (num_classes + 1) 
        voter_count = 0
        for boxes in all_boxes:
            matched_label = None
            for box in boxes:
                dist = np.linalg.norm(np.array([box[1], box[2]]) - np.array(center))
                if dist < 0.01:
                    matched_label = box[0]
                    break
            if matched_label is not None:
                votes[matched_label] += 1
                voter_count += 1

        missing_votes = num_annotators - voter_count
        votes[-1] += missing_votes

        if voter_count >= 2:  # at least have 2 voters
            table.append(votes)
    return np.array(table)

# plot fleiss kappa
def fleiss_vis(kappa_with_penalty, kappa_without_penalty, out_path):
    indices = list(range(len(kappa_with_penalty)))
    plt.figure(figsize=(6.3, 4))
    plt.plot(indices, kappa_with_penalty, 'o-', label="Fleiss' Kappa (penalized)", alpha=0.7)
    plt.plot(indices, kappa_without_penalty, 's--', label="Fleiss' Kappa (unpenalized)", alpha=0.7)

    plt.xlabel('Image Index', fontsize=8)
    plt.ylabel("Fleiss' Kappa", fontsize=8)
    plt.ylim(-0.1, 1.1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'fleiss_comparison.pdf'), dpi=300)
    plt.close()

# plot heatmap
def heatmap_vis(all_boxes, img_size, out_path, filename):
    mask = np.zeros(img_size, dtype=np.float32)
    for annot_boxes in all_boxes:
        for box in annot_boxes:
            _, x_center, y_center, width, height = box
            h, w = img_size
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            mask[y1:y2, x1:x2] += 1
    mask = np.ma.masked_where(mask == 0, mask)
    cmap = plt.cm.jet
    cmap.set_bad(color='white')
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(mask, cmap=cmap, interpolation='nearest')
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True,
                   labelbottom=False, labelleft=False, direction='in', length=6)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
    #fig.colorbar(im, ax=ax)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.linspace(5.0, 25.0, 5))
    fig.savefig(os.path.join(out_path, filename + '.png'), bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

# plot fleiss mean
def fleiss_vis_scatter(fleiss_scores, out_path):
    plt.figure(figsize=(6.3, 4))
    plt.plot(fleiss_scores, 'o-', label="Fleiss' Kappa", alpha=0.7)
    mean_value = np.mean(fleiss_scores)
    plt.axhline(y=mean_value, color='red', linestyle='--', linewidth=2, label=f"Mean Kappa ({mean_value:.2f})")
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Image Index', fontsize=8)
    plt.ylabel("Fleiss' Kappa", fontsize=8)
    plt.title(" ", fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'fleiss_AvH.pdf'), dpi=300)
    plt.close()

def cohen_vis(data, labels, out_path):
    
    fixed_labels = []
    for label in labels:
        a, b = label.split(" vs ")
        if "AI" in [a, b] and a != "AI":
            fixed_labels.append(f"AI vs {a}")
        elif "AI" in [a, b] and b != "AI":
            fixed_labels.append(f"AI vs {b}")
        else:
            fixed_labels.append(f"{min(a, b)} vs {max(a, b)}")

    label_data_pairs = list(zip(fixed_labels, data))
    label_data_pairs.sort(key=lambda x: (not x[0].startswith("AI vs "), x[0]))

    sorted_labels = [x[0] for x in label_data_pairs]
    sorted_data = [x[1] for x in label_data_pairs]
    
    colors = plt.cm.jet(np.linspace(0, 1, len(sorted_labels)))
    fig, ax = plt.subplots(figsize=(6.3, 4))

    box = ax.boxplot(sorted_data, labels=sorted_labels, patch_artist=True, showmeans=True,
                     meanprops={"marker": "*", "markerfacecolor": "red", "markeredgecolor": "red", "markersize": 3})

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for median in box['medians']:
        median.set(color='black', linewidth=2)

    for i, dataset in enumerate(sorted_data):
        y = dataset
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.scatter(x, y, s=100, alpha=0.6, color=colors[i], edgecolors='black')

        mean_val = np.mean(dataset)
        ax.text(i + 1.1, mean_val, f"{mean_val:.2f}", color='red', fontsize=8, va='center')

    ax.set_ylabel("Cohen's Kappa", fontsize=8)
    ax.set_xlabel("Paired AI and HI", fontsize=8)
    ax.set_title(" ", fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'cohen_AvH.pdf'), dpi=300)
    plt.close()

def kappa_matrix(fleiss_scores, pairwise_kappas, filenames, out_path):

    pair_keys = list(pairwise_kappas.keys())
    all_labels = pair_keys + ["Fleiss"]
    
    matrix_all = np.column_stack([pairwise_kappas[k] for k in pair_keys] + [fleiss_scores])
    matrix_all = np.array(matrix_all)

    keep_rows = []
    keep_filenames = []

    for idx in range(matrix_all.shape[0]):
        cohen_vals = matrix_all[idx, :-1]
        num_not_1 = np.sum(np.abs(cohen_vals - 1.0) > 1e-6)
        if num_not_1 > len(cohen_vals) / 2:
            keep_rows.append(matrix_all[idx])
            keep_filenames.append(filenames[idx])

    if not keep_rows:
        print("No samples for plotting")
        return

    matrix = np.vstack(keep_rows)
    masked_matrix = np.ma.masked_where(matrix == 0, matrix)
    
    cmap = plt.cm.jet
    cmap.set_bad(color='white')
    norm = plt.Normalize(vmin=0, vmax=1)

    fig_width_in = 6.3
    fig_height_in = max(3.0, 0.25 * len(keep_filenames))
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))

    im = ax.imshow(masked_matrix, cmap=cmap, norm=norm)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if val > 0:
                color = 'white' if val < 0.5 or val > 0.9 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=color)

    ax.set_xticks(np.arange(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(keep_filenames)))
    ax.set_yticklabels([os.path.splitext(f)[0] for f in keep_filenames])
    ax.set_xlabel("Kappa Type")
    ax.set_ylabel("Image")

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.linspace(0, 1.0, 11))

    fig.tight_layout()
    fig.savefig(os.path.join(out_path, "kappa_matrix.pdf"), dpi=300, format='pdf')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="heatmap plot、Fleiss' and Cohen's Kappa caculation and visulization")
    parser.add_argument("-input", required=True, help="input labels")
    parser.add_argument("-out", required=True, help="output path")
    parser.add_argument("-size", type=int, nargs=2, default=[1440, 1080], help="image size")
    parser.add_argument("-iou", type=float, default=0.5, help="IoU")
    parser.add_argument("-p", action="store_true", help="punish unlabeled objects (default no punishment)")

    args = parser.parse_args()

    annotators = [d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))]
    paths = {annot: os.path.join(args.input, annot) for annot in annotators}
    filenames = set.intersection(*[set(os.listdir(paths[annot])) for annot in annotators])

    kappas, pairwise_kappas = [], {f"{a} vs {b}": [] for a, b in combinations(annotators, 2)}
    os.makedirs(args.out, exist_ok=True)
    kappas_unpenalized = []
    
    for filename in sorted(filenames):
        all_boxes = [read_label(os.path.join(paths[annot], filename)) for annot in annotators]

        fleiss_table = penalized_fleiss_table(all_boxes, iou_threshold=args.iou)
        kappas.append(fleiss_kappa(fleiss_table))
        
        fleiss_table_unpenalized = unpenalized_fleiss_table(all_boxes, iou_threshold=args.iou)
        kappas_unpenalized.append(fleiss_kappa(fleiss_table_unpenalized))

        for (i, j), key in zip(combinations(range(len(annotators)), 2), pairwise_kappas):
            boxes1 = all_boxes[i]
            boxes2 = all_boxes[j]
            matched_indices, unmatched1, unmatched2 = greedy_iou(boxes1, boxes2, iou_threshold=args.iou)

            labels1 = []
            labels2 = []

            for idx1, idx2 in matched_indices:
                labels1.append(boxes1[idx1][0])
                labels2.append(boxes2[idx2][0])

            if args.p:
                for idx in unmatched1:
                    labels1.append(boxes1[idx][0])
                    labels2.append(-1)
                for idx in unmatched2:
                    labels1.append(-1)
                    labels2.append(boxes2[idx][0])
            else:
                pass
            if labels1 and labels2:
                try:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        kappa = cohen_kappa_score(labels1, labels2)
                        if np.isnan(kappa):
                            kappa = 0.0
                except Exception:
                    kappa = 0.0
                pairwise_kappas[key].append(kappa)
            else:
                pairwise_kappas[key].append(0.0)


        heatmap_vis(all_boxes, tuple(args.size), args.out, os.path.splitext(filename)[0])

    fleiss_vis_scatter(kappas, args.out)
    fleiss_vis(kappas, kappas_unpenalized, args.out)
    cohen_vis(
        data=[pairwise_kappas[key] for key in pairwise_kappas],
        labels=[key for key in pairwise_kappas],
        out_path=args.out
    )

    kappa_matrix(kappas, pairwise_kappas, sorted(filenames), args.out)

    pd.DataFrame(pairwise_kappas, index=sorted(filenames)).assign(Fleiss_Kappa=kappas, Fleiss_Kappa_Unpenalized=kappas_unpenalized).to_excel(
        os.path.join(args.out, 'kappa_results.xlsx'))


if __name__ == '__main__':
    main()
