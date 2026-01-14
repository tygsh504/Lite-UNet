import cv2
import numpy as np
import math
import glob
import os
from sklearn.model_selection import train_test_split

class DataLoader():

    def __init__(self, args):
        self.train_data = args.train_data
        self.train_annot = args.train_annot
        self.val_data = args.val_data
        self.val_annot = args.val_annot
        self.image_size = (args.img_width, args.img_height)
        self.batch_size = args.batch_size
        
        # We will count images dynamically in data_generator to ensure filters apply
        self.num_train_imgs = 0
        self.num_val_imgs = 0

    def read_images(self, images_dir):
        imgs = []
        for path in images_dir:
            img = cv2.imread(path)
            
            # --- SAFETY CHECK ---
            if img is None:
                raise ValueError(f"[ERROR] Could not read image file: {path}\n"
                                 f"Please check if the file exists and is a valid image.")
            # --------------------

            img = cv2.resize(img, self.image_size)
            img = (img / 127.5) - 1.0  # Normalize to [-1, 1]
            imgs.append(np.array(img))
        imgs = np.array(imgs)
        return imgs

    def read_masks(self, masks_dir):
        masks = []
        for path in masks_dir:
            graymask = cv2.imread(path, 0)
            
            # --- SAFETY CHECK ---
            if graymask is None:
                raise ValueError(f"[ERROR] Could not read mask file: {path}\n"
                                 f"Please check if the file exists and is a valid image.")
            # --------------------

            graymask = cv2.resize(graymask, self.image_size)
            (_, mask) = cv2.threshold(graymask, 1, 255, cv2.THRESH_BINARY)
            mask = mask / 255.0
            mask = np.expand_dims(mask, axis=-1)
            masks.append(np.array(mask))
        masks = np.array(masks)
        return masks

    def preprocess_masks(self, masks):
        preprocessedMasks = []
        for mask in masks:
            mask = np.expand_dims(mask, axis=-1)
            preprocessedMasks.append(np.array(mask))
        preprocessedMasks = np.array(preprocessedMasks)
        return preprocessedMasks

    def trainDataGenerator(self, imgs_files, masks_files, batch_size=8):
        while True:
            num_batches = math.ceil(len(imgs_files) / batch_size)
            for i in range(0, num_batches):
                current_batch_index = i * batch_size
                # Handle slicing for both regular and last batch automatically
                batch_imgs_files = imgs_files[current_batch_index : current_batch_index + batch_size]
                batch_masks_files = masks_files[current_batch_index : current_batch_index + batch_size]
                
                imgs = self.read_images(batch_imgs_files)
                masks = self.read_masks(batch_masks_files)
                masks = self.preprocess_masks(masks)

                yield (imgs, masks)

    def validationDataGenerator(self, imgs_files, masks_files, batch_size=32):
        while True:
            num_batches = math.ceil(len(imgs_files) / batch_size)
            for i in range(0, num_batches):
                current_batch_index = i * batch_size
                batch_imgs_files = imgs_files[current_batch_index : current_batch_index + batch_size]
                batch_masks_files = masks_files[current_batch_index : current_batch_index + batch_size]
                
                imgs = self.read_images(batch_imgs_files)
                masks = self.read_masks(batch_masks_files)
                masks = self.preprocess_masks(masks)

                yield (imgs, masks)

    def get_train_steps_per_epoch(self):
        return math.ceil((self.num_train_imgs / self.batch_size))

    def get_validation_steps_per_epoch(self):
        return math.ceil((self.num_val_imgs / self.batch_size))
    
    def _get_clean_file_list(self, path):
        """Helper to get sorted list of files, ignoring hidden files like .ipynb_checkpoints"""
        all_files = sorted(glob.glob(os.path.join(path, r"**/*.*"), recursive=True))
        
        # Filter: Keep only valid image extensions and ignore hidden folders
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        clean_files = [
            f for f in all_files 
            if f.lower().endswith(valid_extensions) and '.ipynb_checkpoints' not in f
        ]
        return clean_files

    def data_generator(self):
        # 1. Get clean lists of files
        train_data_dir = self._get_clean_file_list(self.train_data)
        train_annot_dir = self._get_clean_file_list(self.train_annot)
        val_data_dir = self._get_clean_file_list(self.val_data)
        val_annot_dir = self._get_clean_file_list(self.val_annot)
        
        # 2. Update counts based on clean lists
        self.num_train_imgs = len(train_data_dir)
        self.num_val_imgs = len(val_data_dir)
        
        print(f"Found {self.num_train_imgs} training images and {len(train_annot_dir)} training masks.")
        print(f"Found {self.num_val_imgs} validation images and {len(val_annot_dir)} validation masks.")

        # 3. Sanity Check: Mismatch detection
        if self.num_train_imgs != len(train_annot_dir):
            print("WARNING: Mismatch between Training Images and Masks count!")
        if self.num_val_imgs != len(val_annot_dir):
            print("WARNING: Mismatch between Validation Images and Masks count!")

        train_generator = self.trainDataGenerator(train_data_dir, train_annot_dir, self.batch_size)
        validation_generator = self.validationDataGenerator(val_data_dir, val_annot_dir, self.batch_size)

        return train_generator, validation_generator