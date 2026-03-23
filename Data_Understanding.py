import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import seaborn as sns

st.set_page_config(
    layout="wide",
    page_title="Pistachio - Dashboard",
    page_icon=":seedling:",
)

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
    random_state=2026,
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

with st.expander("Image Dataset"):

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

train_ft_16, test_ft_16 = train_test_split(ft_dataset_16, test_size=0.2, random_state=2026, stratify=ft_dataset_16['Class'])

with st.sidebar:  
    # add header  
    st.header("Filters", divider=True)
    # dropdown to select attributes
    selected_attribute = st.selectbox("Attribute: ", train_ft_16.columns[:-1], index=0)
    # multiselect to select species
    selected_class = st.multiselect("Class: ", train_ft_16['Class'].unique())


with st.expander("16-Feature Dataset"):
    st.header("Dataset Overview")
    st.dataframe(train_ft_16.describe(), width='stretch')
    st.header("First 5 rows of the 16-feature dataset:")
    st.dataframe(train_ft_16.head())
    st.header("Class distribution in the 16-feature dataset:")
    st.bar_chart(train_ft_16['Class'].value_counts())

# --------------------------------------------------------
# 28-features

train_ft_28, test_ft_28 = train_test_split(ft_dataset_28, test_size=0.2, random_state=42, stratify=ft_dataset_28['Class'])

with st.expander("28-Feature Dataset"):
    st.header("Dataset Overview")
    st.write(f"28-feature dataset - Training set size: {len(train_ft_28)}, Test set size: {len(test_ft_28)}")
    st.header("First 5 rows of the 28-feature dataset:")
    st.dataframe(train_ft_28.head())
    st.header("Class distribution in the 28-feature dataset:")
    st.bar_chart(train_ft_28['Class'].value_counts())