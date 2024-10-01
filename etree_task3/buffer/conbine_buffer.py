import os, sys
os.chdir(sys.path[0])
import glob
import json
import os.path as osp


if __name__=="__main__":
    dir_path = './'  # 请替换为你的目录路径
    for prefix in ['buffer_bleurt_all', 'buffer_dict_entail_etree_all']:  # 
        with open(osp.join(dir_path, prefix + ".json"), 'r') as f:
            orig_buffer = json.load(f)
        # 使用glob获取所有以 'xx' 开头的文件
        json_files = glob.glob(osp.join(dir_path, 'tmp', f"{prefix}*"))

        for json_file in json_files:
            print(f'Reading file: {json_file}')
            with open(json_file, 'r') as f:
                data = json.load(f)
                orig_buffer.update(data)  # 使用update方法将新的数据添加到merged_dict中
                # for key in data.keys():
                #     if key in orig_buffer:
                #         print(f'Warning: Duplicate key {key} found in {json_file}')
                #         # assert False
                #     orig_buffer[key] = data[key]

            # 将合并的数据写入新的json文件
            with open(osp.join(dir_path, prefix + ".json"), 'w') as outfile:
                json.dump(orig_buffer, outfile)