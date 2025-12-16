import numpy as np
import tensorflow as tf
import keras
from mobilenetv2 import MobileNetV2
from keras import optimizers
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, Activation, BatchNormalization
from keras.layers import UpSampling2D, Input, Concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from keras.metrics import Recall, Precision
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt 
import os # NEW: Import for OS operations
from datetime import datetime # NEW: Import for unique filenames

# Enable mixed precision globally
# mixed_precision.set_global_policy("mixed_float16")

# Custom callback to track the actual learning rate per epoch
class LrHistory(Callback):
    """Callback to log the current learning rate at the start of each epoch."""
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            return
        # Get the current learning rate value
        lr = self.model.optimizer.lr
        if callable(lr):
            # For schedules, evaluate the function
            lr = lr(self.model.optimizer.iterations)
            
        # Ensure 'lr' is in history and append current value
        current_lr = float(tf.keras.backend.get_value(lr))
        # Use history.history.setdefault to safely add 'lr' list if it doesn't exist
        self.model.history.history.setdefault('lr', []).append(current_lr)


class Lite_UNet():
    
    def __init__(self, args):
        self.img_height = args.img_height
        self.img_width = args.img_width
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.epochs = args.epochs
        self.output_dir = args.output_dir

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
        # MobileNetV2 encoder
        model_encoder = MobileNetV2(input_tensor=model_input, weights=weights, include_top=False, alpha=width_muliplier)
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
        # Note: final output forced to float32 for numerical stability

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
        my_callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=self.output_dir,
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=5
            ),
            LrHistory() # Added custom callback to track LR
        ]
        return my_callbacks

    def compile_model(self):
        model = self.build_model()
        base_opt = tf.keras.optimizers.Adam(self.lr, clipnorm=1.0)
        # opt = mixed_precision.LossScaleOptimizer(base_opt)    # <-- wrap optimizer
        opt = base_opt
        metrics = ['accuracy', self.dice_coef, self.iou, Recall(), Precision()]
        model.compile(loss=self.dice_loss, optimizer=opt, metrics=metrics)
        return model

    # MODIFIED: Return history object
    def train(self, train_generator, val_generator, num_train_batches, num_val_batches):
        model = self.compile_model()
        history = model.fit( # Capture the history object
            x=train_generator,
            validation_data=val_generator,
            epochs=self.epochs,
            steps_per_epoch=num_train_batches,
            validation_steps=num_val_batches,
            callbacks=self.define_callbacks()
        )
        return model, history # Return both model and history

    # MODIFIED: Method to plot and SAVE the training history
    def plot_history(self, history):
        """Plots Learning Rate, Training Loss, and Validation Dice Coefficient over epochs and saves the graph."""
        
        # 1. Setup paths - ***THIS SECTION HAS BEEN MODIFIED***
        # The specified absolute path for saving graphs
        graphs_dir = r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Code\Lite-UNet\output_graphs"

        # Create the 'output_graphs' folder if it does not exist
        os.makedirs(graphs_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 5))
        
        # 1. Learning Rate graph
        plt.subplot(1, 3, 1)
        plt.plot(history.history.get('lr', []))
        plt.title('1. Learning Rate over Epochs')
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
        plt.grid(True)
        
        # 2. Train Loss graph
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.title('2. Training Loss over Epochs')
        plt.ylabel('Loss (Dice Loss)')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)

        # 3. Validation Dice Coef graph
        # The custom metric 'dice_coef' is tracked as 'val_dice_coef'
        if 'val_dice_coef' in history.history:
            plt.subplot(1, 3, 3)
            plt.plot(history.history['val_dice_coef'], label='Validation Dice Coef')
            plt.title('3. Validation Dice Coef over Epochs')
            plt.ylabel('Dice Coefficient')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
        else:
            print("Warning: 'val_dice_coef' not found in history. Cannot plot Validation Dice.")
        
        plt.tight_layout()

        # 4. Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_metrics_{timestamp}.png"
        save_path = os.path.join(graphs_dir, filename)
        
        plt.savefig(save_path) # Save the figure instead of showing
        plt.close() # Close the figure to free memory
        
        print(f"Training graphs successfully saved to: {save_path}")