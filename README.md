# CNN Total PyTorch
This repository contains a collection of various libraries and utility functions for deep learning with PyTorch. It is specifically designed for medical tasks.

--- 
## Features
- **Tasks**:
  - **Dataset**:
    - Supports augmentation using **Albumentations**.
    - Provides data preprocessing utilities for medical imaging datasets.
  - **Classification**:
    - Includes predefined architectures for medical image classification.
  - **Segmentation**:
    - Supports semantic and instance segmentation tasks with flexibility for model customization.
  - **Generation**:
    - Contains tools for generating synthetic medical data, useful for data augmentation and balancing.
  - **Training**:
    - Provides **Jupyter Notebook-based training scripts** for ease of use and reproducibility.

--- 
## Installation and Usage
### How to Use
Currently, detailed usage instructions are under development. Custom `.ipynb` files for each task are being created but have not yet been uploaded, as the general usage methodology has not been finalized. If requested, I will upload example `.ipynb` files for demonstration.

For now, the files under the `src/model` directory can be used independently. They return PyTorch `Model` objects (`nn.Module`) and can be directly imported and utilized in your projects.

--- 
> **Note:** 
> 1. A script capable of performing **regression multi-task** is located at `scripts/0_regression_multi_task_pft_xray.ipynb`.
> 2. A script capable of performing **segmentation multi-task** is located at `scripts/1_segmentation_multi_task_dp.py`.
> 3. A script capable of performing **CT-Super resoultion Gan** is located at `scripts/2_ct_super_resolution_gan.py`

--- 
## Why Use This Repository?
Proficiency with this repository allows you to efficiently conduct various tasks and experiments by simply adjusting parameters. It is designed to streamline your workflow and enhance productivity.

--- 
## Limitations
The current documentation is being improved with the addition of comprehensive docstrings to enhance clarity and user guidance.

--- 
## Future Improvements
- Add comprehensive DocStrings.
- Conducting paper work that utilizes the code from this repository.
- I am currently working on implementations for diffusion autoencoder and Med-Seg-Diff.

--- 
## Contact
If you have any questions or suggestions, feel free to reach out via:  
📧 **Email**: tobeor3009@gmail.com  
💬 **GitHub Issues**

--- 

Let me know if you need further refinements! 😊
