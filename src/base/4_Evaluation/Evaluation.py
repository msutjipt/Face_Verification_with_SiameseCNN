import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
import os

def transform_pair_txt(file_path_pair, root_dir, output_path):
    # Read the content of the file again
    with open(file_path_pair, 'r') as file:
        lines = file.readlines()

    # Process each line to create the desired format
    processed_lines = []
    for line in lines:
        parts = line.strip().split('\t')
        png_number_1 = int(parts[1])
        png_number_2 = int(parts[3])
        if len(parts) == 4:
            if (parts[0] == parts[2]):
                new_line = f"{root_dir}/{parts[0]}/{parts[0]}_{parts[1].zfill(4)}.png;{root_dir}/{parts[2]}/{parts[2]}_{parts[3].zfill(4)}.png;0\n"
                processed_lines.append(new_line)
            elif (parts[0] != parts[2]):
                new_line = f"{root_dir}/{parts[0]}/{parts[0]}_{parts[1].zfill(4)}.png;{root_dir}/{parts[2]}/{parts[2]}_{parts[3].zfill(4)}.png;1\n"
                processed_lines.append(new_line)

    # Write the processed lines to a new file
    with open(output_path, 'w') as file:
        file.writelines(processed_lines)

class ImagePairDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame mit den Bildpfaden.
            transform (callable, optional): Optionaler Transform, der auf beide Bilder angewendet wird.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img1_path = self.dataframe.iloc[idx, self.dataframe.columns.get_loc('Image1_Path')]
        img2_path = self.dataframe.iloc[idx, self.dataframe.columns.get_loc('Image2_Path')]

        try:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
        except Exception as e:
            print(f"Error loading images at index {idx}: {e}. Skipping this entry.")
            return self.__getitem__((idx + 1) % len(self))

        label = self.dataframe.iloc[idx, self.dataframe.columns.get_loc('True_Label')]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, img1_path, img2_path, label

class FaceRecognitionModel(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, embedding_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def predict(model, val_dataloader, device, treshold):
    model.eval()
    correct = 0.0
    total = 0.0
    true_labels = []
    predicted_labels = []
    distance = []
    img1_paths = []
    img2_paths = []

    with torch.no_grad():
        for batch_idx, (img1, img2, img1_path, img2_path, labels) in enumerate(val_dataloader):
            labels = labels.view(-1, 1).float()
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)
            img1 = Variable(img1)
            img2 = Variable(img2)
            labels = Variable(labels)

            output1 = model(img1)
            output2 = model(img2)
            euclidean_distance = F.pairwise_distance(output1, output2)
            predicted = torch.tensor([0 if sd < treshold else 1 for sd in euclidean_distance]).to(device)

            predicted_array = predicted.cpu().numpy()
            labels_array = labels.cpu().numpy().flatten().astype(int)
            euclidean_distance = euclidean_distance.cpu().numpy().astype(float)
            zero_matches = np.sum((labels_array == 0) & (predicted_array == 0))
            one_matches = np.sum((labels_array == 1) & (predicted_array == 1))

            true_labels.extend(labels_array)
            predicted_labels.extend(predicted_array)
            distance.extend(euclidean_distance)
            img1_paths.extend(img1_path)
            img2_paths.extend(img2_path)

            total += labels.size(0)
            correct += zero_matches + one_matches
    accuracy = 100 * (correct / total)
    print('Accuracy of the network on the test set: {:.2f}%'.format(accuracy))

    evaluation_df = pd.DataFrame({
    'Image1 Path': img1_paths,
    'Image2 Path': img2_paths,
    'True Label': true_labels,
    'Predicted Label': predicted_labels,
    'Distance': distance })

    return evaluation_df

def main():

    # Step 1: Define all necessary paths

    # For the pair file
    file_path = '.../pairs.txt'
    # Path to store the seperated file
    output_path = '.../pairs_seperated.txt'

    # path to the lfw dataset
    root_dir = '.../lfw_cropped'

    # Path to the store model / model weights
    path_to_stored_model = '.../src/base/5_TrainedModels/ContrastiveLoss/Augmented/siamese_model_contrastive_augmented.pth'

    # Step 2: Call function to transform the pair.txt
    transform_pair_txt(file_path, root_dir, output_path)

    # Step 3: Read processed pair txt as csv file and convert it into a dataframe
    df = pd.read_csv(output_path, delimiter=";", names=['Image1_Path', 'Image2_Path', 'True_Label'])

    # Step 4: Use GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 5: Load stored model

    # If you use the stored model
    siamese_model = torch.load(path_to_stored_model)

    # Uncomment this if you use the model weights
    #siamese_model = FaceRecognitionModel().to(device)
    #siamese_model.load_state_dict(torch.load(path_to_stored_model))

    # Step 6: Call ImagePairDataset to load the lfw images
    transformation = transforms.ToTensor()
    siamese_test_dataset = ImagePairDataset(df, transformation)

    # Step 7: Call the PyTorch DataLoader
    batch_size = 32
    test_dataloader = DataLoader(siamese_test_dataset, shuffle=False, batch_size=batch_size)

    # Step 8: Make Predictions with the stored model. The unscaled Threshold was calculated with the ROC-Curve.
    # How this is done in detail can be found below for the scaled distance. A major

    threshold_unscaled = 0.71
    evaluation_df = predict(siamese_model, test_dataloader, device, threshold_unscaled)

    # Store evaluation_df as a csv file if demanded
    #evaluation_df.to_csv("evaluation_after_prediction.csv", index=False, sep=";")

    # Step 9: Convert columns into a numpy array from the evaluation_df
    true_labels = np.array(evaluation_df['True Label'])
    distances = np.array(evaluation_df['Distance'])

    # Step 10: Use ROC-Curve to determine the unscaled optimal threshold. Here the pos_label=1, because a larger distance
    # indicates a non-match
    fpr, tpr, thresholds = roc_curve(true_labels, distances, pos_label=1)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold_unscaled = thresholds[optimal_idx]
    print(f"The optimal unscaled threshold is: {optimal_threshold_unscaled}")

    # Step 11: Scale the distances on a scala from 0 to 1, where a small distance value indicates a Non-Match and
    # a large distance indicates a Match.

    scaler = MinMaxScaler()
    distances = np.array(evaluation_df['Distance']).reshape(-1, 1)

    scaled_distances = scaler.fit_transform(distances)

    evaluation_df['Distance'] = 1 - scaled_distances
    true_labels = np.array(evaluation_df['True Label'])
    distances = np.array(evaluation_df['Distance'])

    # Step 12: Calculate the optimal threshold for the scaled distance from 0 to 1
    fpr, tpr, thresholds = roc_curve(true_labels, distances, pos_label=0)

    roc_auc = auc(fpr, tpr)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold_scaled = thresholds[optimal_idx]
    print(f"The optimal scaled threshold is: {optimal_threshold_scaled}")

    # Step 14: Adjust the predicted labels with the optimal threshold for plotting the Confusion Matrix
    predicted_labels = [0 if sd > 0.72 else 1 for sd in distances]

    # Step 15: Plot and save the ROC-Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig("ROC.png")

    # Step 16: Plot and save the Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    #plt.show()
    plt.savefig("ConfusionMatrix.png")

    # Step 17: Create and save a mated and non-mated txt-file

    # Only keep the filename of the image
    evaluation_df['Image_1'] = evaluation_df['Image1 Path'].apply(os.path.basename)
    evaluation_df['Image_2'] = evaluation_df['Image2 Path'].apply(os.path.basename)

    # Drop the whole image paths
    evaluation_df.drop(inplace=True, columns=['Image1 Path', 'Image2 Path'])

    # Store the whole evaluation.csv if demanded
    #evaluation_df.to_csv("evaluation_final.csv", index=False, sep=";")

    # Drop columns for predicted label
    evaluation_df.drop(inplace=True, columns=['Predicted Label'])

    # Rename columns distance to probability
    evaluation_df.rename(columns={'Distance': 'Probability'}, inplace=True)

    # Filter dataframes accordingly to their true label. Here 0 is a match and 1 is a non-match!
    mated_df = evaluation_df[evaluation_df['True Label'] == 0]
    non_mated_df = evaluation_df[evaluation_df['True Label'] == 1]

    # Drop true labels, since they are not necessary
    mated_df = mated_df.drop(inplace=False, columns=['True Label'])
    non_mated_df = non_mated_df.drop(inplace=False, columns=['True Label'])

    # Restructure column order
    mated_df = mated_df[['Image_1', 'Image_2', 'Probability']]
    non_mated_df = non_mated_df[['Image_1', 'Image_2', 'Probability']]

    # Save files as csv and as txt
    mated_df.to_csv("mated_images.csv", index=False, sep=";")
    non_mated_df.to_csv("non_mated_images.csv", index=False, sep=";")
    mated_df.to_csv("mated_images.txt", index=False, sep=";")
    non_mated_df.to_csv("non_mated_images.txt", index=False, sep=";")

if __name__ == "__main__":
    main()