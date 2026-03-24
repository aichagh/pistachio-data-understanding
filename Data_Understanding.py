import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import seaborn as sns
import plotly.express as px


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

st.title("Pistachio Dataset - Data Understanding")

tab1, tab2, tab3 = st.tabs(["Image Dataset", "16-Feature Dataset", "28-Feature Dataset"])

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

with tab1:

    st.header("Dataset Overview")
    st.write(f"Training set size: {len(train_subset)}, Test set size: {len(test_subset)}")
    st.write(f"Image shape: {img_dataset[0][0].shape}")
    st.write(f"Class names: {img_dataset.classes}")

    col1, col2, col3, col4, col5 = st.columns(5)
    try:
        batch_images, batch_labels = next(iter(train_loader))
        for col, i in zip([col1, col2, col3, col4, col5], range(5)):
            img = batch_images[i].permute(1, 2, 0).numpy()
            col.image(img, caption=f"Class: {img_dataset.classes[batch_labels[i]]}", width=150)
    except Exception as e:
        st.warning(f"Could not render sample images from DataLoader: {e}")

# ------------------ Feature datasets -------------------
# 16-features

train_ft_16, test_ft_16 = train_test_split(ft_dataset_16, test_size=0.2, random_state=2026, stratify=ft_dataset_16['Class'])

with tab2:
    with st.expander("Dataset Overview", expanded=True):
        selected_classes = st.multiselect("Classes: ", train_ft_16['Class'].unique(), placeholder="Filter by class", default=train_ft_16['Class'].unique())
        filtered_df = train_ft_16[train_ft_16['Class'].isin(selected_classes)]
        st.header("Description of the dataset")
        st.dataframe(filtered_df.describe(), width='stretch')
        st.header("First 5 rows of the 16-feature dataset:")
        st.dataframe(filtered_df.head())
        st.header("Class distribution in the 16-feature dataset:")
        st.bar_chart(train_ft_16['Class'].value_counts())
        
    with st.expander("Visualization", expanded=False):
        selected_attribute = st.selectbox("Select an attribute to visualize: ", train_ft_16.columns[:-1], index=0)
        st.header(f"Distribution of {selected_attribute} by class")
        st.plotly_chart(px.histogram(filtered_df, x=selected_attribute, color='Class', title=f"{selected_attribute} distribution by class", marginal="box"), use_container_width=True)
        
        st.space()
        selected_attribute_a = st.selectbox("Select attribute A for scatter plot: ", train_ft_16.columns[:-1], index=0, key='scatter_a')
        selected_attribute_b = st.selectbox("Select attribute B for scatter plot: ", train_ft_16.columns[:-1], index=1, key='scatter_b')
        st.header(f"Scatter plot of {selected_attribute_a} vs. {selected_attribute_b}")
        fig = px.scatter(filtered_df, x=selected_attribute_a, y=selected_attribute_b, color='Class', title=f"{selected_attribute_a} vs. {selected_attribute_b}")
        st.plotly_chart(fig, use_container_width=True)
        
        
        

# --------------------------------------------------------
# 28-features

train_ft_28, test_ft_28 = train_test_split(ft_dataset_28, test_size=0.2, random_state=2026, stratify=ft_dataset_28['Class'])

with tab3:
    with st.expander("Dataset Overview"):
        selected_classes_28 = st.multiselect("Classes: ", train_ft_28['Class'].unique(), placeholder="Filter by class", default=train_ft_28['Class'].unique())
        filtered_df_2 = train_ft_28[train_ft_28['Class'].isin(selected_classes_28)]
        st.header("Description of the dataset")
        st.write(f"28-feature dataset - Training set size: {len(train_ft_28)}, Test set size: {len(test_ft_28)}")
        st.header("First 5 rows of the 28-feature dataset:")
        st.dataframe(train_ft_28.head())
        st.header("Class distribution in the 28-feature dataset:")
        st.bar_chart(train_ft_28['Class'].value_counts())
    
    
    with st.expander("Visualization", expanded=False):
        selected_attribute_2 = st.selectbox("Select an attribute to visualize: ", train_ft_28.columns[:-1], index=0)
        st.header(f"Distribution of {selected_attribute_2} by class")
        st.plotly_chart(px.histogram(filtered_df_2, x=selected_attribute_2, color='Class', title=f"{selected_attribute_2} distribution by class", marginal="box"), use_container_width=True)
        
        st.space()
        selected_attribute_a_2 = st.selectbox("Select attribute A for scatter plot: ", train_ft_28.columns[:-1], index=0, key='scatter_a_28')
        selected_attribute_b_2 = st.selectbox("Select attribute B for scatter plot: ", train_ft_28.columns[:-1], index=1, key='scatter_b_28')
        st.header(f"Scatter plot of {selected_attribute_a_2} vs. {selected_attribute_b_2}")
        fig = px.scatter(filtered_df_2, x=selected_attribute_a_2, y=selected_attribute_b_2, color='Class', title=f"{selected_attribute_a_2} vs. {selected_attribute_b_2}")
        st.plotly_chart(fig, use_container_width=True)