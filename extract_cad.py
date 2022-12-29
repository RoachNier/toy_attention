'''
    此文件完成了对cad图片宽高、图名的提取
    引用的模型来自PaddleOCR2Pytorch
'''

import numpy as np
import cv2
import time
from collections import defaultdict
# 还需要引用PaddleOCRPytorch的文件，但如何import先暂定
from PaddleOCR2Pytorch.tools.infer.predict_system import patches_recognize

def image2patch(img, h_patch_num=10, v_patch_num=10, h_overlap_ratio=0.0122, v_overlap_ratio=0.008):
    ###### 将一张8k图片拆分为若干个patch, patch之间有交叠
    #### 去掉图片彩色，只留下黑白像素点，已单独写为函数filter_color
    # img = np.sum(img, axis=2)
    # img[np.where(img == 765)] = 255
    # img[np.where(img == 0)] = 0
    # img[np.where(img % 765 != 0)] = 255
    # img = img.astype(np.uint8)
    # img = img[None]
    # img = np.concatenate([img, img, img], axis=0)
    # img = img.transpose(1, 2, 0)
    H, W, C = img.shape
    # 根据patch_num 和 overlap_ratio 进行切片
    patches = {}
    h_over_pix, v_over_pix = int(W * h_overlap_ratio), int(H * v_overlap_ratio)
    h_patch_size = int((W + (h_patch_num - 1) * W * h_overlap_ratio) // h_patch_num)
    v_patch_size = int((H + (v_patch_num - 1) * H * v_overlap_ratio) // v_patch_num)
    for i in range(v_patch_num): 
        for j in range(h_patch_num):
            left_top_x = int(j * h_patch_size - j * h_over_pix)
            left_top_y = int(i * v_patch_size - i * v_over_pix)
            coords = f'{left_top_y}_{left_top_x}'
            #print(left_top_x, left_top_y, v_patch_size, h_patch_size)
            patch = img[left_top_y:left_top_y+v_patch_size, left_top_x:left_top_x+h_patch_size,:]
            patches[coords] = patch
    return patches

def recoginze(patches):
    coord_boxes_text_score_dict = patches_recognize(patches)
    return coord_boxes_text_score_dict

def parse_output(coord_boxes_text_score_dict, image_height, drop_score):
    # 根据内容和坐标位置来确定最大尺寸
    # 最大尺寸通过两个量来判断：一是可能存在的轴网总尺寸(暂时废弃)，二是一定存在的分尺寸；不考虑细尺寸，因为细尺寸往往间隔很小，存在大量的误判
    # coord_boxes_text_score_dict的组成是坐标: [[boxes], [(text, score),...]]
    pix_thresh = 15 / 6224 # 是判断直线的范围，需调节使得直线更准确，但实际使用前应确定标准
    pix_thresh = pix_thresh * image_height
    coord_text_dict = {}
    coordy_text_dict = defaultdict(list)
    coord_list = []
    min_coord_x_delta = 0
    for coord, bts in coord_boxes_text_score_dict.items(): 
        box_list, text_score_list = bts
        for box, text_score in zip(box_list, text_score_list):
            #print(f'{box}***********{text_score[0]}')
            center = [(box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4, (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4]
            left_top_y, left_top_x = int(coord.split('_')[0]), int(coord.split('_')[1])
            center[0] += left_top_x
            center[1] += left_top_y
            center[0] = int(center[0])
            center[1] = int(center[1])
            text = text_score[0]
            score = text_score[1]
            if score >= drop_score and is_number(text) and int(text) >= 100: # 这里的100是为了去掉图纸中的圆圈可能识别出的数
                min_coord_x_delta = max(min_coord_x_delta, (abs(box[1][0]-box[0][0]) + abs(box[3][0] - box[2][0])) / 2 / 2) # 多除了2，表示取最大数字宽度的一半
                 # if already_included(center, coord_list, pix_thresh): continue # 去掉重复的数字识别结果，因为有过滤操作，所以关掉此行
                 # 在追加识别结果时增加了过滤的操作，主要过滤的是局部被识别的数字
                coord_list, coord_text_dict, coordy_text_dict = filter_append(coord_list, coord_text_dict, coordy_text_dict, center, int(text), min_coord_x_delta, pix_thresh)
    # 根据得到的coordy_text_dict，进行数字提取，首先进行y坐标相同的数字合并
    coordy_digit_tuples = sorted(coordy_text_dict.items())
    # 用快慢指针简化合并数组的运算
    i, j = 0, 1
    while j != len(coordy_digit_tuples):
        if abs(coordy_digit_tuples[i][0] - coordy_digit_tuples[j][0]) <= pix_thresh:
            for item in coordy_digit_tuples[j][1]:
                coordy_digit_tuples[i][1].append(item)
        else :
            i += 1
            coordy_digit_tuples[i] = coordy_digit_tuples[j]
        j += 1
    coordy_digit_tuples = coordy_digit_tuples[: i + 1]
    # for item in coordy_digit_tuples:
    #     print(item)
    # 如果有最外层的标注
    line_max, single_max = None, None 
    if len(coordy_digit_tuples[0][1]) == 1 or len(coordy_digit_tuples[-1][1]) == 1:
        y_upline, y_downline = sum(coordy_digit_tuples[1][1]), sum(coordy_digit_tuples[-2][1])
        line_max = max(y_upline, y_downline)
        print(f'用直线判断所得值：{line_max}')
        up_max, down_max = coordy_digit_tuples[0][1], coordy_digit_tuples[-1][1]
        single_max = max(up_max, down_max)
        print(f'且最外层总标注所得值：{single_max[0]}')
    else:
        y_upline, y_downline = sum(coordy_digit_tuples[0][1]), sum(coordy_digit_tuples[-1][1])
        line_max = max(y_upline, y_downline)
        print(f'用直线判断所得值：{line_max}')
    if single_max and single_max[0] >= line_max:
        return single_max[0]
    else:
        return line_max

def process_pipe(img, h_patch_num=10, v_patch_num=10, h_overlap_ratio=0.0122, v_overlap_ratio=0.008, drop_score=0.5):
    img = color_filter(img)
    H, W, C = img.shape
    img90 = rotate(img)
    patches = image2patch(img, h_patch_num=h_patch_num, v_patch_num=v_patch_num, h_overlap_ratio=h_overlap_ratio, v_overlap_ratio=v_overlap_ratio)
    patches90 = image2patch(img90, h_patch_num=h_patch_num, v_patch_num=v_patch_num, h_overlap_ratio=h_overlap_ratio, v_overlap_ratio=v_overlap_ratio)
    coord_boxes_text_score_dict = recoginze(patches)
    coord_boxes_text_score_dict90 = recoginze(patches90)
    result = parse_output(coord_boxes_text_score_dict, H, drop_score)
    result90 = parse_output(coord_boxes_text_score_dict90, W, drop_score)
    width_height = [result, result90]
    print(result, result90)
    return width_height

# utils
def color_filter(img):
    img = np.sum(img, axis=2)
    img[np.where(img == 765)] = 255
    img[np.where(img == 0)] = 0
    img[np.where(img % 765 != 0)] = 255
    img = img.astype(np.uint8)
    img = img[None]
    img = np.concatenate([img, img, img], axis=0)
    img = img.transpose(1, 2, 0)
    return img

def rotate(image):
    img90 = np.rot90(image, 1)
    #print(f'image size {image.shape}, img90.size {img90.shape}')
    return img90

def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        pass
    return False

def already_included(center, center_list, thresh):
    for item in center_list:
        if abs(center[0] - item[0]) <= thresh and abs(center[1] - item[1]) <= thresh:
            return True
    return False

def filter_append(coord_list, coord_text_dict, coordy_text_dict, center, text, min_coord_x_delta, min_coord_y_delta):
    exchange_flag = False # 是否经过了替换，如果是，则所有的dict需要重写
    add_flag = True # 是否需要直接添加
    for coord, existed_text in coord_text_dict.items():
        existed_center_x, existed_center_y = int(coord.split('_')[0]), int(coord.split('_')[1])
        if abs(center[0] - existed_center_x) < min_coord_x_delta and abs(center[1] - existed_center_y) < min_coord_y_delta: #如果识别结果中有一个是局部的切片, 则保留数值大的那一个
            if existed_text < text:
                pop_item = coord_text_dict.pop(coord)
                print(f"删除了局部被识别的数字，值为{existed_text},中心点坐标为{coord},替换的值为{text}，中心坐标为{center[0]}_{center[1]}")
                coord_text_dict[f'{center[0]}_{center[1]}'] = int(text)
                exchange_flag = True
                add_flag = False
                break
            else:
                add_flag = False # 如果已存在的值更大，则无需处理
    if add_flag:
        coord_list.append([center[0], center[1]])
        coord_text_dict[f'{center[0]}_{center[1]}'] = int(text)
        coordy_text_dict[center[1]].append(int(text))
    if exchange_flag:
        # 重新生成dict和list
        coord_list.clear()
        coordy_text_dict.clear()
        for coord, existed_text in coord_text_dict.items():
            coord = [int(coord.split('_')[0]), int(coord.split('_')[1])]
            coordy = coord[1]
            coord_list.append(coord)
            coordy_text_dict[coordy].append(existed_text)
    return coord_list, coord_text_dict, coordy_text_dict


if __name__ == '__main__':
    img = cv2.imread('/home/aiserver/projects/CAD_OCR/test_imgs/202212191154542158.png')
    time0 = time.time()
    wh = process_pipe(img)
    time1 = time.time()
    print(f'完成长宽检测总时间为:{time1 - time0}')


