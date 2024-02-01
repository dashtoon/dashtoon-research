import pandas as pd
import os
import sys
import numpy as np

file_path = ""


def misalignment_penalization_score(mapping_graph, a):
    penalty = np.abs(a)
    fmap_freq = {}

    for _, map_list in mapping_graph.items():
        mlist = map_list[0]
        blist = map_list[1]

        l1 = len(mlist)
        l2 = len(blist)

        if l1==0 and l2==0:
            penalty += 1
        else:
            for item in mlist:
                if item in fmap_freq:
                    fmap_freq[item] += 1
                else:
                    fmap_freq[item] = 1
            for item in blist:
                if item in fmap_freq:
                    fmap_freq[item] += 1
                else:
                    fmap_freq[item] = 1

            penalty += (l1+l2-1)

    for freq in fmap_freq.values():
        penalty += (freq - 1)

    return penalty  


def main():

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path)
    print("Input file read..!")

    df["fname"] = df["fname"].apply(lambda x: str(x).zfill(5))

    df['penalty'] = df.apply(lambda row: misalignment_penalization_score(row['sf_map_info'], row['difference_in_num_of_masks']), axis = 1)
    df = df.sort_values(by = 'penalty', ascending=True)
    print("Penalty scores calculation done for all the image pairs..!")

    if file_path.endswith(".csv"):
        df.to_csv(file_path, index = False)
    elif file_path.endswith(".json"):
        df.to_json(file_path, orient="records", compression="infer")
        
    print("File updated with penalty scores saved..!")


if __name__ == "__main__":
    main()
