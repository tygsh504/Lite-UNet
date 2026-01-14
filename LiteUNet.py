# import numpy as np
# import tensorflow as tf
# import keras
# from mobilenetv2 import MobileNetV2
# from keras import optimizers
# from keras.models import load_model
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, Activation, BatchNormalization
# from keras.layers import UpSampling2D, Input, Concatenate
# from keras.models import Model
# from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
# from keras.metrics import Recall, Precision
# from tensorflow.keras import mixed_precision
# import matplotlib.pyplot as plt 
# import os 
# from datetime import datetime 
# import pandas as pd # NEW: Import pandas for Excel handling

# # Enable mixed precision globally
# # mixed_precision.set_global_policy("mixed_float16")

# # Custom callback to track the actual learning rate per epoch
# class LrHistory(Callback):
#     """Callback to log the current learning rate at the start of each epoch."""
#     def on_epoch_begin(self, epoch, logs=None):
#         if not hasattr(self.model.optimizer, 'lr'):
#             return
#         # Get the current learning rate value
#         lr = self.model.optimizer.lr
#         if callable(lr):
#             # For schedules, evaluate the function
#             lr = lr(self.model.optimizer.iterations)
            
#         # Ensure 'lr' is in history and append current value
#         current_lr = float(tf.keras.backend.get_value(lr))
#         # Use history.history.setdefault to safely add 'lr' list if it doesn't exist
#         self.model.history.history.setdefault('lr', []).append(current_lr)


# class Lite_UNet():
    
#     def __init__(self, args):
#         self.img_height = args.img_height
#         self.img_width = args.img_width
#         self.batch_size = args.batch_size
#         self.lr = args.lr
#         self.epochs = args.epochs
#         self.output_dir = args.output_dir
#         # NEW: capture the result directory from arguments
#         self.result_dir = args.result_dir 
        
#         # Ensure the directory exists
#         if self.result_dir:
#             os.makedirs(self.result_dir, exist_ok=True)

#     def build_model(self, width_muliplier=0.35, weights="imagenet"):
#         def decoder_block(x, residual, n_filters, n_conv_layers=2):
#             up = UpSampling2D((2, 2))(x)
#             merge = Concatenate()([up, residual])
            
#             x = Conv2D(n_filters, (3, 3), padding="same")(merge)
#             x = BatchNormalization()(x)
#             x = Activation("relu")(x)
#             for i in range(n_conv_layers-1): 
#                 x = Conv2D(n_filters, (3, 3), padding="same")(x)
#                 x = BatchNormalization()(x)
#                 x = Activation("relu")(x)
#             return x

#         def get_encoder_layers(encoder, concat_layers, output_layer):
#             return [encoder.get_layer(layer).output for layer in concat_layers], encoder.get_layer(output_layer).output

#         model_input = Input(shape=(self.img_height, self.img_width, 3), name="input_img")
#         # MobileNetV2 encoder
#         model_encoder = MobileNetV2(input_tensor=model_input, weights=weights, include_top=False, alpha=width_muliplier)
#         concat_layers, encoder_output = get_encoder_layers(
#             model_encoder,
#             ["input_img", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu", "block_13_expand_relu"],
#             "block_16_expand_relu"
#         )

#         filters = [3, 48, 48, 96, 192]
#         x = encoder_output

#         for layer_name, n_filters in zip(concat_layers[::-1], filters[::-1]):
#             x = decoder_block(x, layer_name, n_filters)

#         out = Conv2D(1, (1, 1), padding="same", activation="sigmoid", dtype="float32")(x)
        
#         model = Model(model_input, out)
#         return model

#     def iou(self, y_true, y_pred):
#         smooth = 1e-6
#         y_true = tf.keras.layers.Flatten()(y_true)
#         y_pred = tf.keras.layers.Flatten()(y_pred)
#         intersection = tf.reduce_sum(y_true * y_pred)
#         union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
#         return (intersection + smooth) / (union + smooth)

#     def dice_coef(self, y_true, y_pred):
#         smooth = 1e-6
#         y_true = tf.keras.layers.Flatten()(y_true)
#         y_pred = tf.keras.layers.Flatten()(y_pred)
#         intersection = tf.reduce_sum(y_true * y_pred)
#         return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

#     def dice_loss(self, y_true, y_pred):
#         return 1.0 - self.dice_coef(y_true, y_pred)

#     def define_callbacks(self):
#         # Determine where to save the best model checkpoint
#         # If result_dir is set, save 'best_model.h5' there, otherwise use output_dir
#         if self.result_dir:
#             checkpoint_path = os.path.join(self.result_dir, "best_model_weights.h5")
#         else:
#             checkpoint_path = self.output_dir

#         my_callbacks = [
#             keras.callbacks.ModelCheckpoint(
#                 filepath=checkpoint_path,
#                 monitor='val_loss',
#                 mode='min',
#                 save_best_only=True,
#                 verbose=True
#             ),
#             keras.callbacks.ReduceLROnPlateau(
#                 monitor='val_loss', factor=0.2, patience=5
#             ),
#             LrHistory() 
#         ]
#         return my_callbacks

#     def compile_model(self):
#         model = self.build_model()
#         base_opt = tf.keras.optimizers.Adam(self.lr, clipnorm=1.0)
#         opt = base_opt
#         metrics = ['accuracy', self.dice_coef, self.iou, Recall(), Precision()]
#         model.compile(loss=self.dice_loss, optimizer=opt, metrics=metrics)
#         return model

#     def train(self, train_generator, val_generator, num_train_batches, num_val_batches):
#         model = self.compile_model()
#         history = model.fit( 
#             x=train_generator,
#             validation_data=val_generator,
#             epochs=self.epochs,
#             steps_per_epoch=num_train_batches,
#             validation_steps=num_val_batches,
#             callbacks=self.define_callbacks()
#         )
#         return model, history 

#     def plot_history(self, history):
#         """Plots Learning Rate, Training Loss, and Validation Dice Coefficient over epochs and saves the graph."""
        
#         # Use the result_dir if provided, otherwise default
#         if self.result_dir:
#             graphs_dir = self.result_dir
#         else:
#             graphs_dir = r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Code\Lite-UNet\output_graphs"

#         os.makedirs(graphs_dir, exist_ok=True)
        
#         plt.figure(figsize=(15, 5))
        
#         # 1. Learning Rate graph
#         plt.subplot(1, 3, 1)
#         plt.plot(history.history.get('lr', []))
#         plt.title('1. Learning Rate over Epochs')
#         plt.ylabel('Learning Rate')
#         plt.xlabel('Epoch')
#         plt.grid(True)
        
#         # 2. Train Loss graph
#         plt.subplot(1, 3, 2)
#         plt.plot(history.history['loss'], label='Train Loss')
#         plt.title('2. Training Loss over Epochs')
#         plt.ylabel('Loss (Dice Loss)')
#         plt.xlabel('Epoch')
#         plt.legend()
#         plt.grid(True)

#         # 3. Validation Dice Coef graph
#         if 'val_dice_coef' in history.history:
#             plt.subplot(1, 3, 3)
#             plt.plot(history.history['val_dice_coef'], label='Validation Dice Coef')
#             plt.title('3. Validation Dice Coef over Epochs')
#             plt.ylabel('Dice Coefficient')
#             plt.xlabel('Epoch')
#             plt.legend()
#             plt.grid(True)
        
#         plt.tight_layout()

#         # Save the figure
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"training_graphs_{timestamp}.png"
#         save_path = os.path.join(graphs_dir, filename)
        
#         plt.savefig(save_path) 
#         plt.close() 
        
#         print(f"Training graphs successfully saved to: {save_path}")

#     # NEW FUNCTION: Save metrics to Excel
#     def save_results_to_excel(self, history):
#         """Saves all training metrics from history to an Excel file."""
        
#         if self.result_dir:
#             save_dir = self.result_dir
#         else:
#             # Fallback if no directory specified
#             save_dir = r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Code\Lite-UNet\output_graphs"
            
#         os.makedirs(save_dir, exist_ok=True)

#         # Create DataFrame from history dictionary
#         df = pd.DataFrame(history.history)
        
#         # Add an 'epoch' column at the beginning (1-based index)
#         df.insert(0, 'epoch', range(1, len(df) + 1))
        
#         # Construct filename
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"training_metrics_{timestamp}.xlsx"
#         save_path = os.path.join(save_dir, filename)
        
#         # Save to Excel
#         try:
#             df.to_excel(save_path, index=False)
#             print(f"Training metrics successfully saved to Excel: {save_path}")
#         except Exception as e:
#             print(f"Error saving Excel file: {e}")

import numpy as np
import tensorflow as tf
import os
import pandas as pd  # Required for Excel saving
import matplotlib.pyplot as plt
from datetime import datetime

# Import the wrapper from your local file
from mobilenetv2 import MobileNetV2

# Use tensorflow.keras for stability in Colab
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, 
    Activation, BatchNormalization, UpSampling2D, Input, Concatenate
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
)
from tensorflow.keras.metrics import Recall, Precision

# Custom callback to track the actual learning rate per epoch
class LrHistory(Callback):
    """Callback to log the current learning rate at the start of each epoch."""
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            return
        lr = self.model.optimizer.lr
        if callable(lr):
            lr = lr(self.model.optimizer.iterations)
        current_lr = float(tf.keras.backend.get_value(lr))
        self.model.history.history.setdefault('lr', []).append(current_lr)

class Lite_UNet():
    
    def __init__(self, args):
        self.img_height = args.img_height
        self.img_width = args.img_width
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.epochs = args.epochs
        self.output_dir = args.output_dir
        
        # --- Capture result directory ---
        # Defaults to 'results' if argument not provided
        self.result_dir = getattr(args, 'result_dir', 'results')
        if self.result_dir:
            os.makedirs(self.result_dir, exist_ok=True)

    def build_model(self, width_muliplier=0.35, weights="imagenet"):
        def decoder_block(x, residual, n_filters, n_conv_layers=2):
            up = UpSampling2D((2, 2))(x)
            merge = Concatenate()([up, residual])
            
            x = Conv2D(n_filters, (3, 3), padding="same")(merge)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            for i in range(n_conv_layers-1): 
                x = Conv2D(n_filters, (3, 3), padding="same")(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)
            return x

        def get_encoder_layers(encoder, concat_layers, output_layer):
            return [encoder.get_layer(layer).output for layer in concat_layers], encoder.get_layer(output_layer).output

        model_input = Input(shape=(self.img_height, self.img_width, 3), name="input_img")
        
        # Calls the MobileNetV2 from your local mobilenetv2.py (which now wraps the built-in one)
        model_encoder = MobileNetV2(
            input_tensor=model_input, 
            weights=weights, 
            include_top=False, 
            alpha=width_muliplier
        )
        
        concat_layers, encoder_output = get_encoder_layers(
            model_encoder,
            ["input_img", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu", "block_13_expand_relu"],
            "block_16_expand_relu"
        )

        filters = [3, 48, 48, 96, 192]
        x = encoder_output

        for layer_name, n_filters in zip(concat_layers[::-1], filters[::-1]):
            x = decoder_block(x, layer_name, n_filters)

        out = Conv2D(1, (1, 1), padding="same", activation="sigmoid", dtype="float32")(x)
        
        model = Model(model_input, out)
        return model

    def iou(self, y_true, y_pred):
        smooth = 1e-6
        y_true = tf.keras.layers.Flatten()(y_true)
        y_pred = tf.keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        return (intersection + smooth) / (union + smooth)

    def dice_coef(self, y_true, y_pred):
        smooth = 1e-6
        y_true = tf.keras.layers.Flatten()(y_true)
        y_pred = tf.keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

    def dice_loss(self, y_true, y_pred):
        return 1.0 - self.dice_coef(y_true, y_pred)

    def define_callbacks(self):
        # Save best model to result_dir if available, else output_dir
        if self.result_dir:
            checkpoint_path = os.path.join(self.result_dir, "best_model.h5")
        else:
            checkpoint_path = self.output_dir

        my_callbacks = [
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=5
            ),
            LrHistory()
        ]
        return my_callbacks

    def compile_model(self):
        model = self.build_model()
        base_opt = tf.keras.optimizers.Adam(self.lr, clipnorm=1.0)
        metrics = ['accuracy', self.dice_coef, self.iou, Recall(), Precision()]
        model.compile(loss=self.dice_loss, optimizer=base_opt, metrics=metrics)
        return model

    def train(self, train_generator, val_generator, num_train_batches, num_val_batches):
        model = self.compile_model()
        history = model.fit(
            x=train_generator,
            validation_data=val_generator,
            epochs=self.epochs,
            steps_per_epoch=num_train_batches,
            validation_steps=num_val_batches,
            callbacks=self.define_callbacks()
        )
        return model, history

    def plot_history(self, history):
        save_dir = self.result_dir if self.result_dir else "output_graphs"
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 5))
        
        # 1. Learning Rate
        plt.subplot(1, 3, 1)
        plt.plot(history.history.get('lr', []))
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.grid(True)
        
        # 2. Loss
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)

        # 3. Dice Coefficient
        plt.subplot(1, 3, 3)
        if 'val_dice_coef' in history.history:
            plt.plot(history.history['val_dice_coef'], label='Val Dice')
        if 'dice_coef' in history.history:
            plt.plot(history.history['dice_coef'], label='Train Dice')
        plt.title('Dice Coefficient')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"training_graphs_{timestamp}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Graphs saved to: {save_path}")

    def save_results_to_excel(self, history):
        save_dir = self.result_dir if self.result_dir else "results"
        os.makedirs(save_dir, exist_ok=True)

        # Convert history dict to DataFrame
        df = pd.DataFrame(history.history)
        # Insert epoch column (1-based)
        df.insert(0, 'epoch', range(1, len(df) + 1))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"training_metrics_{timestamp}.xlsx")
        
        try:
            df.to_excel(save_path, index=False)
            print(f"Metrics saved to Excel: {save_path}")
        except Exception as e:
            print(f"Error saving Excel: {e}")