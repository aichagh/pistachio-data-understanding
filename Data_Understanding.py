import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import seaborn as sns

st.set_page_config(layout="wide")

# loading feature datasets
ft_dataset_16 = pd.read_excel('./Pistachio_Image_Dataset/Pistachio_Image_Dataset/Pistachio_16_Features_Dataset/Pistachio_16_Features_Dataset.xlsx')
ft_dataset_28 = pd.read_excel('./Pistachio_Image_Dataset/Pistachio_Image_Dataset/Pistachio_28_Features_Dataset/Pistachio_28_Features_Dataset.xlsx')

# loading image datasets with transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
img_dataset = datasets.ImageFolder('./Pistachio_Image_Dataset/Pistachio_Image_Dataset/Pistachio_Image_Dataset', transform=transform)

# ------------------ Image dataset -------------------

# Create train and test sets (80/20 split)
img_indices = list(range(len(img_dataset)))
train_indices, test_indices = train_test_split(
    img_indices,
    test_size=0.2,
    random_state=42,
    stratify=img_dataset.targets,
)

# Create Subset objects
train_subset = Subset(img_dataset, train_indices)
test_subset = Subset(img_dataset, test_indices)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

st.title("Pistachio Dataset - Data Understanding")

st.write("## Image Dataset")

st.write("### Dataset Overview")
st.write(f"Training set size: {len(train_subset)}, Test set size: {len(test_subset)}")
st.write(f"Class names: {img_dataset.classes}")

# show first batch from train_loader
try:
    batch_images, batch_labels = next(iter(train_loader))
    st.write(f"Batch size: {batch_images.shape[0]}")
    fig, ax = plt.subplots(1, 5, figsize=(20, 20))
    for i in range(5):
        img = batch_images[i].permute(1, 2, 0).numpy()
        ax[i].imshow(img)
        ax[i].set_title(f"Class: {img_dataset.classes[batch_labels[i]]}")
        ax[i].axis('off')
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Could not render sample images from DataLoader: {e}")

# ------------------ Feature datasets -------------------
# 16-features

train_ft_16, test_ft_16 = train_test_split(ft_dataset_16, test_size=0.2, random_state=42, stratify=ft_dataset_16['Class'])


st.write("## 16-Feature Dataset")
st.write("### Dataset Overview")
st.write(f"16-feature dataset - Training set size: {len(train_ft_16)}, Test set size: {len(test_ft_16)}")
st.write("### First 5 rows of the 16-feature dataset:")
st.dataframe(train_ft_16.head())
st.write("### Class distribution in the 16-feature dataset:")
st.bar_chart(train_ft_16['Class'].value_counts())
def plot_feature_grid(df, title, n_cols=4):
    feature_columns = df.columns[:-1]
    n_features = len(feature_columns)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for idx, feature in enumerate(feature_columns):
        ax = axes[idx]
        ax.hist(df[feature], bins=20, alpha=0.7)
        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')

    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    st.write(f"### {title}")
    st.pyplot(fig)

st.write("### Feature distributions in the 16-feature dataset:")
plot_feature_grid(train_ft_16, "16-feature distributions")
st.write("### Correlation heatmap for the 16-feature dataset:")
plt.figure(figsize=(12, 10))
correlation_matrix = train_ft_16.drop(columns=['Class']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap - 16-feature dataset")
st.pyplot(plt)
st.write("### Most highly correlated feature pairs in the 16-feature dataset (>= 0.8 absolute correlation):")
highly_correlated_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            highly_correlated_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

for pair in highly_correlated_pairs:
    st.write(f"- {pair[0]} and {pair[1]} ({correlation_matrix.loc[pair[0], pair[1]]:.2f})")

# --------------------------------------------------------
# 28-features

train_ft_28, test_ft_28 = train_test_split(ft_dataset_28, test_size=0.2, random_state=42, stratify=ft_dataset_28['Class'])

st.write("## 28-Feature Dataset")
st.write("### Dataset Overview")
st.write(f"28-feature dataset - Training set size: {len(train_ft_28)}, Test set size: {len(test_ft_28)}")
st.write("### First 5 rows of the 28-feature dataset:")
st.dataframe(train_ft_28.head())
st.write("### Class distribution in the 28-feature dataset:")
st.bar_chart(train_ft_28['Class'].value_counts())
st.write("### Feature distributions in the 28-feature dataset:")
plot_feature_grid(train_ft_28, "28-feature distributions")