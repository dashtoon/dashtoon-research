import os
import pandas as pd
import sys
# import torch.distributed as dist
import PIL.Image
import io
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import os
from datasets import Dataset, concatenate_datasets, Features, Image, Value, load_from_disk

import matplotlib.pyplot as plt
import cv2
from lang_segment_anything.lang_sam.lang_sam import LangSAM
from lang_segment_anything.lang_sam.utils import draw_image
from argparse import ArgumentParser
import json

save_annotations = True
output_annotation_dir ='/mnt/data1/naman/projects/storyboard_cleanup/outputs/annotated_images_as_pair'
calculate_masks_overlap = True
resize_to_same_size = True
chunk_execution = True
save_in_batches = True
save_every = 100

overlap_threshold = 0.60
box_overlap_threshold = 0.65


def get_dataframe(frame_path):

    df = pd.read_csv(frame_path)
    df["fname"] = df["fname"].apply(lambda x: str(x).zfill(5))
    return  df


def get_dataset(path):

    dataset = load_from_disk(path)
    return dataset


def get_annotated_image(image, masks, boxes, phrases, logits):

    labels = [
        f"{phrase}, score: {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    annotated_image_array = draw_image(np.array(image), masks, boxes, labels)
    annotated_image_array = cv2.resize(annotated_image_array, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

    pil_image = PIL.Image.fromarray(annotated_image_array).convert('RGB')

    return pil_image, annotated_image_array


def get_single_annotated_image(storyboard_annotated_image_array, final_annotated_image_array):
    
    img = np.concatenate([storyboard_annotated_image_array, final_annotated_image_array], axis = 1)
    pil_img = PIL.Image.fromarray(img).convert('RGB')
    return pil_img


def get_predictions(model, image, prompt):

    masks, boxes, phrases, logits = model.predict(image, prompt)
    return masks, boxes, phrases, logits


def get_mask_overlap_score(mask1, mask2):

    binary_mask1 = mask1.numpy().astype(np.uint8)
    binary_mask2 = mask2.numpy().astype(np.uint8)

    # binary_mask1 = cv2.resize(binary_mask1, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    # binary_mask2 = cv2.resize(binary_mask2, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

    intersection = np.logical_and(binary_mask1, binary_mask2).sum()
    union = np.logical_or(binary_mask1, binary_mask2).sum()

    if union == 0:
        return 0.0

    iou = intersection / union

    return iou


def get_box_overlap_score(box1, box2):

    # Calculate intersection area
    x_intersection = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    y_intersection = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    intersection_area = x_intersection * y_intersection

    # Calculate union area
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / max(union_area, 1e-10)  # Avoid division by zero

    return iou


def masks_overlaping_calculations(storyboard_masks, final_masks, storyboard_boxes, final_boxes):

    s2f = {}
    s2f_box = {}
    multi_smask_mapping = 0
    multi_sbox_mapping = 0
    s2f_overlaps = 0

    overlap_scores = []
    box_overlap_scores = []

    sf_map_info = {}

    for sidx, storyboard_mask in enumerate(storyboard_masks):
        multi_map = 0
        box_multimap = 0
        box_global_multimap = False
        global_multimap = False
        mask_sublist, box_sublist = [], []
        for fidx, final_mask in enumerate(final_masks):
            overlap_score = get_mask_overlap_score(storyboard_mask, final_mask) 
            # print('mask', type(overlap_score), overlap_score)
            if overlap_score >= overlap_threshold:

                mask_sublist.append(fidx)

                overlap_scores.append(overlap_score)
                s2f_overlaps+=1
                if storyboard_mask in s2f:
                    multi_map+=1
                    if multi_map > 1:
                        global_multimap = True
                else:
                    s2f[storyboard_mask] = final_mask
                    multi_map = 1
            else:
                box_overlap_score = get_box_overlap_score(storyboard_boxes[sidx], final_boxes[fidx])
                # print('box', type(np.float64(box_overlap_score.item())), box_overlap_score)
                if box_overlap_score >= box_overlap_threshold:

                    box_sublist.append(fidx)

                    box_overlap_scores.append(np.float64(box_overlap_score.item()))
                    s2f_overlaps+=1
                    if sidx in s2f_box:
                        box_multimap+=1
                        if box_multimap > 1:
                            box_global_multimap = True
                    else:
                        s2f_box[sidx] = fidx
                        box_multimap = 1
                # else:
                #     box_overlap_scores.append(-1)
                # overlap_scores.append(-1)
        if global_multimap:
            multi_smask_mapping+=1
        if box_global_multimap:
            multi_sbox_mapping+=1

        sf_map_info[sidx] = [mask_sublist, box_sublist]

    return s2f_overlaps, multi_smask_mapping, multi_sbox_mapping, overlap_scores, box_overlap_scores, sf_map_info


def execute(dataset, model, storyboard_prompt, final_prompt, df, output_json = None, file_output_path = None):

    num_storyboard_masks, num_final_masks = [], []
    num_multimaps_s2f, num_multimaps_s2f_box = [], []
    s2f_total_overlaps = []
    all_overlap_scores, all_box_overlap_scores = [], []
    f_names = []
    all_sf_map_info = []

    fname_set = {}
    if save_in_batches:
        if len(output_json) > 0:
            fname_set = set(pd.DataFrame(output_json).f_name.tolist())

    for idx, sample in enumerate(tqdm(dataset)):

        fname = sample["fname"]

        if save_in_batches:
            if fname in fname_set:
                continue

        f_names.append(fname)
        storyboard_image = sample['storyboard_image']
        final_image = sample['final_image']

        caption = sample['caption']
        
        if resize_to_same_size:
            storyboard_image = PIL.Image.fromarray(cv2.resize(np.array(storyboard_image), 
                                                            dsize = (800, 800), 
                                                            interpolation=cv2.INTER_CUBIC)).convert('RGB')
            final_image = PIL.Image.fromarray(cv2.resize(np.array(final_image), 
                                                            dsize = (800, 800), 
                                                            interpolation=cv2.INTER_CUBIC)).convert('RGB')

        storyboard_masks, storyboard_boxes, storyboard_phrases, storyboard_logits = get_predictions(model, 
                                                                                                    storyboard_image,
                                                                                                    storyboard_prompt)
        final_masks, final_boxes, final_phrases, final_logits = get_predictions(model,
                                                                                final_image,
                                                                                final_prompt)

        ## annotate images with bounding boxes and segmentation 
        if save_annotations:
            _, storyboard_annotated_image_array = get_annotated_image(storyboard_image,
                                                                    storyboard_masks,
                                                                    storyboard_boxes,
                                                                    storyboard_phrases,
                                                                    storyboard_logits)
            
            _, final_annotated_image_array = get_annotated_image(final_image,
                                                                final_masks,
                                                                final_boxes,
                                                                final_phrases,
                                                                final_logits)
            ## save the annotated storyboard and final image as a single image
            final_single_annotated_image = get_single_annotated_image(storyboard_annotated_image_array, final_annotated_image_array)
            final_single_annotated_image.save(f"{output_annotation_dir}/{fname}.jpg")

        num_storyboard_masks.append(storyboard_masks.shape[0])
        num_final_masks.append(final_masks.shape[0])

        if calculate_masks_overlap:
            s2f_overlaps, num_multi_maps_s2f, num_multi_maps_s2f_box, overlap_scores, box_overlap_scores, sf_map_info  = masks_overlaping_calculations(
                                                                                                    storyboard_masks, 
                                                                                                    final_masks,
                                                                                                    storyboard_boxes,
                                                                                                    final_boxes
                                                                                                    ) 
            num_multimaps_s2f.append(num_multi_maps_s2f)
            num_multimaps_s2f_box.append(num_multi_maps_s2f_box)
            s2f_total_overlaps.append(s2f_overlaps)
            all_overlap_scores.append(overlap_scores)
            all_box_overlap_scores.append(box_overlap_scores)
            all_sf_map_info.append(sf_map_info)

        if save_in_batches:
            output_dict = {
                'f_name': fname,
                'num_storyboard_masks': storyboard_masks.shape[0],
                'num_final_masks': final_masks.shape[0],
                'difference_in_num_of_masks': storyboard_masks.shape[0] - final_masks.shape[0],
                'total_overlaps_s2f': s2f_overlaps,
                'num_multimaps_s2f': num_multi_maps_s2f,
                'num_multimaps_s2f_box': num_multi_maps_s2f_box,
                'overlap_scores': overlap_scores,
                'box_overlap_scores': box_overlap_scores,
                'sf_map_info': sf_map_info
            }
            output_json.extend([output_dict])
            if (idx + 1) % save_every == 0:
                with open(file_output_path, 'w') as fp:
                    json.dump(output_json, fp, indent = 3)
                print(f'output json updated by {save_every} more rows..!')

        # if idx>=23:
        #     break

    if save_in_batches:
        with open(file_output_path, 'w') as fp:
            json.dump(output_json, fp, indent = 3)
        print('output json file updated')

    # print(num_storyboard_masks)
    # print(num_final_masks)
    # print(s2f_total_overlaps)
    # print(num_multimaps_s2f)
    # print(num_multimaps_s2f_box)
    # print(all_overlap_scores)
    # print(all_box_overlap_scores)
    # sys.exit(0)

    if chunk_execution:
        df_chunk = pd.DataFrame()
        df_chunk['fname'] = f_names
        df_chunk['num_storyboard_masks'] = num_storyboard_masks
        df_chunk['num_final_masks'] = num_final_masks
        df_chunk['difference_in_num_of_masks'] = df_chunk['num_storyboard_masks'] - df_chunk['num_final_masks']
        df_chunk['total_overlaps_s2f'] = s2f_total_overlaps     
        df_chunk['num_multimaps_s2f'] = num_multimaps_s2f
        df_chunk['num_multimaps_s2f_box'] = num_multimaps_s2f_box
        df_chunk['overlap_scores'] = all_overlap_scores
        df_chunk['box_overlap_scores'] = all_box_overlap_scores
        df_chunk['sf_map_info'] = all_sf_map_info
    else:
        df['num_storyboard_masks'] = num_storyboard_masks
        df['num_final_masks'] = num_final_masks
        df['difference_in_num_of_masks'] = df['num_storyboard_masks'] - df['num_final_masks']
        df['total_overlaps_s2f'] = s2f_total_overlaps     
        df['num_multimaps_s2f'] = num_multimaps_s2f
        df['num_multimaps_s2f_box'] = num_multimaps_s2f_box
        df['overlap_scores'] = all_overlap_scores
        df['box_overlap_scores'] = all_box_overlap_scores
        df['sf_map_info'] = all_sf_map_info

    if save_in_batches and chunk_execution:
        return df_chunk, 'fail_safe'
    if chunk_execution:
        return df_chunk, 'none'
    return df, 'none'


def main(args):

    dataframe_path = '/mnt/data1/naman/projects/storyboard_cleanup/data/storyboard_master_data_2.csv'
    dataset_path = f'/mnt/data1/naman/projects/storyboard_cleanup/data/storyboard_hf_dataset_chunks_54753/storyboard_hf_dataset_chunk_{args.chunk}'
    dataframe_output_path = '/mnt/data1/naman/projects/storyboard_cleanup/outputs'

    # dist.init_process_group('nccl')

    model = LangSAM()
    storyboard_text_prompt = 'person'
    final_text_prompt = 'person'
    
    print(dataset_path)

    df = get_dataframe(dataframe_path)
    dataset = get_dataset(dataset_path)

    if chunk_execution:
        file_output_path = f"{dataframe_output_path}/hf_storyboard_dataset_output_chunk_{dataset_path.split('_')[-1]}.json"
    else:
        file_output_path = f"{dataframe_output_path}/hf_stroyboard_dataset_output.json"

    if not os.path.exists(file_output_path):
        with open(file_output_path, 'w') as fp:
            json.dump([], fp, indent=3)
    with open(file_output_path, 'r') as fp:
        output_json = json.load(fp)
    print('output json file loaded')

    df_final, failsafe = execute(dataset, model, storyboard_text_prompt, final_text_prompt, df, output_json, file_output_path)

    if failsafe != 'fail_safe':
        df_final.to_json(file_output_path, orient='records', compression='infer')
        print('output file saved.')
    else:
        df_final.to_json(f"{file_output_path.split('.')[0]}_fail_safe.json", orient='records', compression='infer')
        print('fail_safe output file saved')

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument(
        "--gpu_id",
        type = int,
        default = 0,
        choices = [0,1,2,3],
        help = ""
    )

    parser.add_argument(
        "--chunk",
        type = int,
        default = 1,
        choices = [1,2,3,4],
        help = ""
    )

    args = parser.parse_args()

    main(args)
  

        
