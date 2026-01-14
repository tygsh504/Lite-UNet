from Dataloader import DataLoader
from LiteUNet import Lite_UNet
import argparse

parser = argparse.ArgumentParser(description='Lite-UNet training script.')

## Arguments for Dataloader for Bacterial Leaf Blight
parser.add_argument('--train_data',default=r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Combined\Training_Ori"
                    ,type=str,help='Training images path')
parser.add_argument('--train_annot',default=r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Combined\Training_GT"
                    ,type=str,help='Training masks path')
parser.add_argument('--val_data',default=r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Combined\Val_Ori"
                    ,type=str,help='Validation images path')
parser.add_argument('--val_annot',default=r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Combined\Val_GT"
                    ,type=str,help='Validation masks path')

parser.add_argument('--img_width',default=384,type=int)
parser.add_argument('--img_height',default=512,type=int)

# Arguments for lite-UNet model
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training.')
parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate for optimizers.')
parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs.')
parser.add_argument('--output_dir', default='Lite-UNet_model_combined.h5', type=str, help='Path for saving the model after training.')

# NEW ARGUMENT: Directory to store Excel results and Graphs
parser.add_argument('--result_dir', default=r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Code\Lite-UNet\results", 
                    type=str, help='Directory to store graphs and excel results.')

args = parser.parse_args()

dl=DataLoader(args) 
train_gen,val_gen=dl.data_generator()
train_steps=dl.get_train_steps_per_epoch()
val_steps=dl.get_validation_steps_per_epoch()

LiteUNet=Lite_UNet(args)
# Capture the model and history object returned by train 
model, history = LiteUNet.train(train_gen,val_gen,train_steps,val_steps)

# Plot and save the training history
LiteUNet.plot_history(history)

# NEW: Save metrics to Excel
LiteUNet.save_results_to_excel(history)