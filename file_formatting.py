import json
import os
import re
import shutil


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("orig_path", type=str, help="the high res imeges")
    parser.add_argument("dataset_path", type=str, help="where the coco.json is located")
    parser.add_argument("dst_path", type=str, help="where you want the output")
    args = parser.parse_args()
    low_res_path = args.dataset_path
    high_q_dir_path = args.orig_path
    new_high_q_dir_path = args.dst_path
    # first copy the data to a new file
    safe_make_dir(new_high_q_dir_path)
    flatten_to_dir(high_q_dir_path, new_high_q_dir_path)
    coco = json.load(open(os.path.join(low_res_path, "_annotations.coco.json")))
    high_q_dir = set(os.listdir(new_high_q_dir_path))
    shutil.copyfile(os.path.join(low_res_path, "_annotations.coco.json"),
                    os.path.join(new_high_q_dir_path, "_annotations.coco.json"))
    check_dir = set(os.listdir(new_high_q_dir_path))
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
