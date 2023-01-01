#!/usr/bin/python
import colorsys
import os
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import argparse
import mrcnn.model as modellib
from mrcnn import visualize
import sys
import tensorflow as tf
from numpy import random
import cv2
from segmentor_utils import init_config, TRAIN_IMAGE_SIZE

sys.path.append(r'C:\Users\yotam\Desktop\yotam\python\Phenomics_PreProcessing_Scripts\Relevent_for_yotam\LeafSegmentor2')
from segmentor_utils import *

# # Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# # Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))



def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors



def Detect_leaves(from_folder,to_folder):
    parser = argparse.ArgumentParser(description = "Description for my parser")
    parser.add_argument("-H", "--Help", help = "Help argument", required = False, action='store_true')
    parser.add_argument("-m", "--model", help = "Model argument", required = False, default = "")
    parser.add_argument("-f", "--from_folder", help = "Image input folder argument", required = False, default = "")
    parser.add_argument("-t", "--to_folder", help = "Image output folder argument", required = False, default = "")
    parser.add_argument("-g", "--ground_truth_folder", help="Ground truth folder argument", required=False, default="")
    parser.add_argument("-o", "--object", help = "Object argument", required = False, action='store_true')

    argument = parser.parse_args()
    status = False

    from_model = r'C:\Users\yotam\Desktop\yotam\python\Phenomics_PreProcessing_Scripts\Relevent_for_yotam\LeafSegmentor2\models\banana_intel_0200_07_09.h5'
    subfolder_out = 'objects/'
    maskOrImageFlag = False

    # AZ insertion for IoU validation
    total_score = 0  # average IoU on all validation images
    GROUND_TRUTH_DIR = "./GT_MASKS/"
    GROUND_TRUTH_FLAG = False
    GROUND_TRUTH_MIN_SIZE_COEFF = 0.05  # 0.03    0.05


    if argument.Help:
        print("Help:")
        print("For Help (this one) use '-H' or '--Help'")
        print("To change default model '-m' or '--model' with path to h5 file")
        print("To change input folder '-f' or '--from_folder' with path to input folder of images")
        print("To change output model '-t' or '--to_folder' with path to output folder of images")
        print("To change ground truth masks folder '-g' or '--ground_truth_folder' with path to ground truth folder of masks")
        print("To output binary masks instead of objects '-o' or '--object'")

        status = True
        exit()
    if argument.model:
        print("You have used '-m' or '--model' with argument: {0}".format(argument.model))
        from_model = argument.model
        status = True
    if argument.from_folder:
        print("You have used '-f' or '--from' with argument: {0}".format(argument.from_folder))
        from_folder = argument.from_folder
        status = True
    if argument.to_folder:
        print("You have used '-t' or '--to' with argument: {0}".format(argument.to_folder))
        to_folder = argument.to_folder
        status = True
    if argument.ground_truth_folder:
        print("You have used '-g' or '--ground_truth' with argument: {0}".format(argument.ground_truth_folder))
        GROUND_TRUTH_DIR = argument.ground_truth_folder
        GROUND_TRUTH_FLAG = True
        status = True
    if argument.object:
        print("You have used '-o' or '--object' with argument: {0}".format(argument.object))
        maskOrImageFlag = True
        status = True
    if not status:
        print("Maybe you want to use -H or -m or -f or -t or -g or -o as arguments ?")


    inference_config = init_config(name="leaves", image_size=TRAIN_IMAGE_SIZE, images_per_GPU=1, inference=True)

    list_of_images = [os.path.join(from_folder, f) for f in sorted(os.listdir(from_folder)) if os.path.isfile(os.path.join(from_folder, f))]

    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
    #model_path = model.find_by_name(from_model)
    print("Loading weights from ", from_model)
    model.load_weights(from_model, by_name=True)

    os.makedirs(os.path.dirname(to_folder), exist_ok=True)
    os.makedirs(os.path.dirname(to_folder+'/'+subfolder_out+'/'), exist_ok=True)

    if GROUND_TRUTH_FLAG:
        score_file = open(to_folder + '/' + "scores_" + str(GROUND_TRUTH_MIN_SIZE_COEFF) + ".txt", "w")
        subfolder_validation = 'validation/'
        os.makedirs(os.path.dirname(to_folder + '/' + subfolder_validation + '/'), exist_ok=True)

    for current_file in list_of_images:
    
        print(current_file)

        current_image = cv2.cvtColor(cv2.imread(current_file), cv2.COLOR_BGR2RGB)
        current_image = Image.fromarray(current_image)

        current_image = np.array(current_image)[:,:,0:3]
    
        results = model.detect([current_image], verbose=1)
        r = results[0]

        _,_,l = r['masks'].shape
        print(l)
        colors = random_colors(l)

        image_name = current_file.split("/")[-1]
        if os.name == "nt":
            image_name = image_name.split("\\")[-1]
        print(image_name)

        for i in range(l):
            mask = r['masks'][:,:,i]

            # AZ - save mask contour to file
            tmp_contour = measure.find_contours(mask, 0.5)
            if not os.path.exists(to_folder+'//'+image_name):
                os.makedirs(to_folder+'//'+image_name)
            np.savetxt(to_folder+'//'+image_name+'//'+str(i)+'.txt',tmp_contour[0])

            if maskOrImageFlag:
                image = Image.fromarray(mask.astype('uint8')*255)
            else:
                idx=(mask==1)
                imr=np.ones(np.shape(current_image))
                imr[idx]=current_image[idx]
                image = Image.fromarray(imr.astype('uint8'), 'RGB')
            save_to = to_folder + '/' + subfolder_out + '/' + image_name.split('.')[0] + '_' + str(i).zfill(3) + '.png'
            image.save(save_to,"PNG")

        #visualize.display_instances(current_image, r['rois'], r['masks'], r['class_ids'], ['BG', 'leaf'], r['scores'])
        #visualize.save_instances(current_image, r['rois'], r['masks'], r['class_ids'], ['BG', 'leaf'], r['scores'], figsize=np.shape(current_image)[0:2], ax=get_ax(), colors=colors, save_to=os.path.join(to_folder, image_name))

        if GROUND_TRUTH_FLAG:
            # AZ start validation of single image
            # TODO - log/results file

            # get ground truth masks for this image
            # note: this should be done only once for each validation image (if train, do it once at the beginning, not after each epoch).
            image_name_prefix = image_name.split(".")[0] + "_GT_"
            num_gt_masks = 0
            h = current_image.shape[0]
            w = current_image.shape[1]
            gt_min_size = GROUND_TRUTH_MIN_SIZE_COEFF * GROUND_TRUTH_MIN_SIZE_COEFF * h * w
            gt_file_names = []
            for root, dirs, files in os.walk(GROUND_TRUTH_DIR):
                for file in files:
                    if file.startswith(image_name_prefix):
                        # read GT file, and use the GT only if num_pixels in mask > Threshold
                        tmp = np.array(Image.open(GROUND_TRUTH_DIR + file))
                        tmp_size = np.count_nonzero(tmp)
                        if tmp_size > gt_min_size:
                            gt_file_names.append(file)
                            num_gt_masks = num_gt_masks + 1
                            print(file)

            gt_masks = np.zeros([h,w,num_gt_masks])
            for i in range(num_gt_masks):
                curr_gt_file = GROUND_TRUTH_DIR + gt_file_names[i]
                curr_mask = np.array(Image.open(curr_gt_file))
                gt_masks[:,:,i] = curr_mask

            # create empty IoU matrix M (num_ground_truth_masks x num detected_masks)
            # note: if validation during training - this should be done after each epoch.
            all_iou = np.zeros(shape=[num_gt_masks,l])

            # fill IoU matrix
            # for each mask m1 in ground truth
            #   for each mask m2 in detected
            #       M(m1,m2) = IoU(m1,m2)
            for i in range(num_gt_masks):
                mask_i = gt_masks[:,:,i]
                for j in range(l):
                    mask_j = r['masks'][:,:,j]
                    intersection = np.logical_and(mask_i,mask_j)
                    union = np.logical_or(mask_i,mask_j)
                    numI = np.count_nonzero(intersection)
                    numU = np.count_nonzero(union)
                    all_iou[i,j] = numI/numU

            # calculate total (or average) IoU
            # curr_score = 0
            # for each mask m1 in ground truth
            #   find mask m2 in detected with highest Iou(m1,m2)
            #   remove m1 from list of ground truth
            #   remove m2 from list of detected
            #   curr_score = curr_score + IoU(m1,m2)
            curr_score = 0
            for i in range(num_gt_masks):
                # find max value and indices of max value
                max_iou = np.amax(all_iou)
                curr_score = curr_score + max_iou
                max_idx = np.argmax(all_iou)
                max_idx_row, max_idx_col = divmod(max_idx, all_iou.shape[1])

                # save image of GT and detected masks
                # if max_iou > 0:
                #     # TODO - if IoU==0, display only gt mask
                #     idx1 = (gt_masks[:,:,max_idx_row] == 1)
                #     idx2 =  (r['masks'][:,:,max_idx_col] == 1)
                #     idx_intersection = np.logical_and(idx1, idx2)
                #
                #     # imr = np.zeros(np.shape(current_image))
                #     # imr[idx1] = [0, 0, 128]
                #     # imr[idx2] = [0, 128, 0]
                #     # imr[idx_intersection] = [128, 0,0 ]
                #     # image = Image.fromarray(imr.astype('uint8'), 'RGB')
                #
                #     imr = current_image.astype(np.uint32).copy()
                #     alpha = 0.1
                #     color_gt = [255,0,0]
                #     color_detected = [0,0,255]
                #     for c in range(3):
                #         imr[:, :, c] = np.where(gt_masks[:,:,max_idx_row] == 1, imr[:, :, c] * (1 - alpha) + alpha * color_gt[c], imr[:, :, c])
                #     for c in range(3):
                #         imr[:, :, c] = np.where(r['masks'][:,:,max_idx_col] == 1, imr[:, :, c] * (1 - alpha) + alpha * color_detected[c], imr[:, :, c])
                #     image = Image.fromarray(imr.astype('uint8'), 'RGB')
                #     save_to = to_folder + '/' + subfolder_validation + '/' + image_name.split('.')[0] + '_' + str(max_idx_row+1).zfill(3) + '_' + str(max_idx_col+1).zfill(3) + '.png'
                #     image.save(save_to, "PNG")

                # remove row/col of max value (set zeros)
                for j in range(all_iou.shape[1]):
                    all_iou[max_idx_row,j] = 0
                for j in range(all_iou.shape[0]):
                    all_iou[j,max_idx_col] = 0

            if num_gt_masks > 0:
                curr_score = curr_score / num_gt_masks
            else:
                curr_score = 1
            total_score = total_score + curr_score
            score_file.write(str(curr_score))
            score_file.write('\n')

            print("IoU score: " + str(curr_score))
            print("finished image: " + image_name)
            # AZ end validation of single image

    if GROUND_TRUTH_FLAG:
        total_score = total_score / len(list_of_images)
        print("average IoU scores: " + str(total_score))
        score_file.write('\n')
        score_file.write(str(total_score))
        score_file.close()



if __name__ == "__main__":

    from_folder = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Images\Patched_Pics\640-640'
    to_folder = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Annotations\Leaves\640-640_Patched_Annotations'
    Detect_leaves(from_folder,to_folder)


