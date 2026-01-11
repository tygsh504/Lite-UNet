import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Added for plotting
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import argparse

# ==========================================
# 1. DEFINE METRICS & LOSSES
# ==========================================
def iou(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# ==========================================
# 2. FLOPS UTILITY
# ==========================================
def get_flops(model):
    try:
        concrete = tf.function(lambda inputs: model(inputs))
        concrete_func = concrete.get_concrete_function(
            [tf.TensorSpec([1, *inputs.shape[1:]], inputs.dtype) for inputs in model.inputs]
        )
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            return flops.total_float_ops
    except Exception:
        return 0

# ==========================================
# 3. TESTING SCRIPT
# ==========================================
def run_test(args):
    # Setup Directories
    output_folder = "testing_output"
    vis_folder = os.path.join(output_folder, "visualizations") # New folder for images
    os.makedirs(vis_folder, exist_ok=True)
    
    print(f"Loading model from: {args.model_path}")
    
    # Load Model
    model = load_model(args.model_path, custom_objects={
        'dice_loss': dice_loss, 
        'dice_coef': dice_coef, 
        'iou': iou
    })

    input_shape = model.input_shape[1:3] # (height, width)
    
    # Calculate Params & FLOPs
    total_params = model.count_params()
    flops = get_flops(model)
    print(f"Params: {total_params:,} | FLOPs: {flops:,}")

    # Prepare Data
    test_img_paths = sorted(glob.glob(os.path.join(args.test_data, r"**/*.*"), recursive=True))
    test_mask_paths = sorted(glob.glob(os.path.join(args.test_annot, r"**/*.*"), recursive=True))

    if len(test_img_paths) == 0:
        print("No images found! Check paths.")
        return

    results_list = []
    m_prec = Precision()
    m_recall = Recall()

    print(f"Processing {len(test_img_paths)} images...")

    for i, (img_path, mask_path) in enumerate(zip(test_img_paths, test_mask_paths)):
        filename = os.path.basename(img_path)
        
        # 1. Read & Preprocess Image
        original_img = cv2.imread(img_path)
        img = cv2.resize(original_img, (input_shape[1], input_shape[0]))
        
        # Normalize for model [-1, 1]
        img_norm = (img / 127.5) - 1.0 
        img_batch = np.expand_dims(img_norm, axis=0)

        # 2. Read & Preprocess Mask
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (input_shape[1], input_shape[0]))
        (_, mask) = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1)
        mask_batch = np.expand_dims(mask, axis=0)

        # 3. Predict
        pred_batch = model.predict(img_batch, verbose=0)
        pred_mask = (pred_batch > 0.5).astype(np.float32)

        # 4. Save Visualization (Evaluated Image)
        # ---------------------------------------------------------
        # We need to denormalize the image back to [0, 255] for display
        # Formula: (x + 1) * 127.5
        display_img = (img_norm + 1.0) * 127.5
        display_img = display_img.astype(np.uint8)
        # Convert BGR (OpenCV default) to RGB (Matplotlib default)
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        # Prepare masks for display (0 or 1 -> 0 or 255)
        display_mask = (mask * 255).astype(np.uint8)
        display_pred = (pred_mask[0] * 255).astype(np.uint8)

        # Create the plot
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.title(f"Original: {filename}")
        plt.imshow(display_img)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(display_mask, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(display_pred, cmap='gray')
        plt.axis('off')

        # Save
        save_path = os.path.join(vis_folder, f"eval_{filename}.png")
        plt.savefig(save_path)
        plt.close() # Close to free memory
        # ---------------------------------------------------------

        # 5. Calculate Metrics
        y_true_f = mask.flatten()
        y_pred_f = pred_mask.flatten()
        
        intersection = np.sum(y_true_f * y_pred_f)
        union_iou = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
        union_dice = np.sum(y_true_f) + np.sum(y_pred_f)
        smooth = 1e-6

        dice_val = (2. * intersection + smooth) / (union_dice + smooth)
        iou_val = (intersection + smooth) / (union_iou + smooth)
        acc_val = np.sum(y_true_f == y_pred_f) / len(y_true_f)

        m_prec.reset_state()
        m_recall.reset_state()
        m_prec.update_state(mask_batch, pred_mask)
        m_recall.update_state(mask_batch, pred_mask)
        prec_val = m_prec.result().numpy()
        recall_val = m_recall.result().numpy()

        if (prec_val + recall_val) > 0:
            f1_val = 2 * (prec_val * recall_val) / (prec_val + recall_val)
        else:
            f1_val = 0.0

        results_list.append({
            "Filename": filename,
            "Dice": dice_val, "IoU": iou_val,
            "Precision": prec_val, "Recall": recall_val,
            "Accuracy": acc_val, "F1_Score": f1_val
        })

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_img_paths)}...")

    # Save Excel Results
    df = pd.DataFrame(results_list)
    avg_metrics = df.mean(numeric_only=True)
    
    summary_data = {
        "Metric": ["Dice", "IoU", "Precision", "Recall", "Accuracy", "F1", "Params", "FLOPs"],
        "Value": [
            avg_metrics["Dice"], avg_metrics["IoU"], avg_metrics["Precision"], 
            avg_metrics["Recall"], avg_metrics["Accuracy"], avg_metrics["F1_Score"],
            total_params, flops
        ]
    }
    
    excel_path = os.path.join(output_folder, "evaluation_results.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
        df.to_excel(writer, sheet_name="Detailed", index=False)
    
    print(f"\nDone! Images saved to: {vis_folder}")
    print(f"Excel saved to: {excel_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='Lite-UNet_model_combined.h5')
    
    # UPDATE THESE PATHS TO YOUR DATA
    default_test_img = r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Tungro\Infer_Ori"
    default_test_mask = r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Tungro\Infer_GT"

    parser.add_argument('--test_data', type=str, default=default_test_img)
    parser.add_argument('--test_annot', type=str, default=default_test_mask)
    
    run_test(parser.parse_args())