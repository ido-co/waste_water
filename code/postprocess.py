import json
import numpy as np

import pandas as pd


def calc_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def postprocess_results(annotations, predictions):
    cat_dict = {}
    with open(annotations) as d:
        input = json.load(d)
        cat_dict = {}
        for dct in input['categories']:
            cat_dict[dct['name']] = dct['id']
    open_floc = cat_dict['open_floc']
    spherical_floc = cat_dict['spherical_floc']

    with open(predictions) as c:
        output = json.load(c)
    output_df = pd.DataFrame.from_dict(output, orient='columns')
    alt = pd.read_json(predictions)
    output_df['drop'] = np.ones(len(output_df), dtype=bool)
    output_df['index'] = range(len(output_df))
    output_df.groupby(['image_id']).apply(lambda x: drop_overlapping(x, output_df, spherical_floc, open_floc))

    filtered = output_df[output_df['drop']]
    print(f"filtered {len(output_df) - len(filtered)} annotations")
    filtered = filtered.drop(['drop', 'index'], axis=1)
    with open("filtered.json", "w") as outfile:
        filtered.to_json(path_or_buf=outfile, orient='records')
    return output_df


def drop_overlapping(image_df, df, drop, ref):
    drop_df = image_df[image_df.category_id == drop]
    ref_df = image_df[image_df.category_id == ref]
    if drop_df.empty or ref_df.empty:
        return
    for drop_index, drop_row in drop_df.iterrows():
        for ref_index, ref_row in ref_df.iterrows():
            if drop_index == ref_index:
                continue
            if calc_iou(drop_row.bbox, ref_row.bbox) > 0.25:
                df.loc[drop_index, 'drop'] = False


if __name__ == '__main__':
    hook = []
    output_df = postprocess_results("../data/file_out/_annotations.coco.json", "../data/coco_instances_results.json")
    import cv2, temp

    # i = 2
    # cv2.imshow(f"xmpl {i}", temp.visualize_by_dict(output_df[~output_df['drop']].iloc[i].to_dict()))
    for i, row in output_df[~output_df['drop']].iterrows():
        temp.visualize_by_dict(row.to_dict(),f"image_{i}.png")
