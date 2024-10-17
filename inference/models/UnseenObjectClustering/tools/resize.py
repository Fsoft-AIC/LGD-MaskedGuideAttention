import cv2

import cv2
import numpy as np

# # Create a grayscale depth mask
img = cv2.imread('/home/tuanvovan/Documents/grasp-amodal/UnseenObjectClustering/data/demo/000010-color.png')
device = "cuda"
# depth_mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Adjust depth (example: increase contrast for better depth effect)
# depth_mask = cv2.equalizeHist(depth_mask)
# print(depth_mask)
# cv2.imwrite('/home/tuanvovan/Documents/grasp-amodal/UnseenObjectClustering/data/demo/000010-depth.png',depth_mask)
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["vit_h"](checkpoint="/home/tuanvovan/Documents/grasp-amodal/sam_vit_h_4b8939.pth")
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(img)
print(masks)