import json
import os
import re
import shutil
import argparse
import contextlib


def safe_make_dir(name, basepath=''):
    dir_path = os.path.join(basepath, name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def flatten_to_dir(path, new_dir):
    for file in os.listdir(path):
        n_path = os.path.join(path, file)
        if os.path.isdir(n_path):
            flatten_to_dir(n_path, new_dir)
        elif re.search(r"\.png$", file):
            shutil.copyfile(n_path, os.path.join(new_dir, file))


"""
copy the images from given path and puts them with the new path with the annotation coco file of the   
to use , then split tje json file to train test eval on a ratio of 0.6:0.2:0.2 
"""
if __name__ == '__main__':
    # low_res_path = r"C:\Users\Administrator\PycharmProjects\ml_workshop\wastwatwer_data\Waste_Water_2.v21-for_cs_workshop_v2.coco\train"
    # high_q_dir_path = r"C:\Users\Administrator\PycharmProjects\ml_workshop\wastwatwer_data\wastwatwer"
    # new_high_q_dir_path = "wastewater_all_out_ver"
    # first copy the data to a new file
    parser = argparse.ArgumentParser()
    parser.add_argument("low_res_path", type=str, help="src file path")
    parser.add_argument("high_q_dir_path", type=str, help="path to the original images", nargs='?',
                        default=os.path.join(os.getcwd(), "..", "data", "wastwatwer_orig_img", ))
    parser.add_argument("new_path", type=str, help="output path ")
    args = parser.parse_args()
    low_res_path = args.low_res_path
    high_q_dir_path = args.high_q_dir_path
    new_high_q_dir_path = args.new_path

    # copy the data to a new file
    safe_make_dir(new_high_q_dir_path)
    flatten_to_dir(high_q_dir_path, new_high_q_dir_path)
    coco = json.load(open(os.path.join(low_res_path, "_annotations.coco.json")))
    high_q_dir = set(os.listdir(new_high_q_dir_path))
    shutil.copyfile(os.path.join(low_res_path, "_annotations.coco.json"),
                    os.path.join(new_high_q_dir_path, "_annotations.coco.json"))
    check_dir = set(os.listdir(new_high_q_dir_path))
    working_dir = os.getcwd()
    os.chdir(new_high_q_dir_path)
    for img in coco['images']:
        name = re.findall("[a-zA-Z0-9]+_\d+_\d+_\d+_[a-zA-Z0-9]+_\d+", img['file_name'])[0] + ".png"
        if name in check_dir:
            os.rename(name, img['file_name'])
            # print(f"copyed {name}")
        else:
            print(f"skipped {img['file_name']}")

    print('-' * 10 + ('-' * 10))
    print('-' * 10 + f'number of files copyed = {len(os.listdir("./"))}')
    print('-' * 10 + 'done' + ('-' * 10))
    os.chdir(working_dir)
    print(10 * "=" + "spliting data into train test eval" + 10 * "=")
    annon_path = os.path.join(new_high_q_dir_path, "_annotations.coco.json")
    tmp_path = os.path.join(new_high_q_dir_path, "ww_data_trainng")
    tmp_path_2 = os.path.join(new_high_q_dir_path, "ww_data_test_eval")
    test_eval_path = tmp_path_2 + ".json"

    os.system(f"pyodi coco random-split {annon_path} {tmp_path} --val-percentage 0.4 ")
    shutil.move(tmp_path + "_train.json", os.path.join(new_high_q_dir_path, "ww_data_train.json"))
    shutil.move(tmp_path + "_val.json", test_eval_path)

    os.system(f"pyodi coco random-split {test_eval_path} {tmp_path_2} --val-percentage 0.5 ")
    shutil.move(tmp_path_2 + "_train.json", os.path.join(new_high_q_dir_path, "ww_data_eval.json"))
    shutil.move(tmp_path_2 + "_val.json", os.path.join(new_high_q_dir_path, "ww_data_test.json"))
    os.remove(test_eval_path)

    print(15 * "=")
