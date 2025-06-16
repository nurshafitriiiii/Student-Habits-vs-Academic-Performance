# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mne  # pip install mne

# Set your project base path
base_path = r'C:\Users\User\IdeaProjects\student_hvsa\plots'

# Define full paths for image files
img1_path = os.path.join(base_path, 'categorical_distributions.png')
img2_path = os.path.join(base_path, 'correlation_heatmap.png')
img3_path = os.path.join(base_path, 'numerical_distributions.png')

# Create the MNE report
report = mne.Report(title='Student Dataset Visual Report')

# Function to add image to the report
def add_image_to_report(img_path, title, section):
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title)
        report.add_figure(fig=fig, title=title, section=section)
        plt.close(fig)
    else:
        print(f"[ERROR] File not found: {img_path}")

# Add each image to the report
add_image_to_report(img1_path, 'Categorical Distributions', 'Data Visualization')
add_image_to_report(img2_path, 'Correlation Heatmap', 'Data Visualization')
add_image_to_report(img3_path, 'Numerical Distributions', 'Data Visualization')

# Save the report as HTML
report.save(fname=os.path.join(base_path, 'student_report.html'), overwrite=True)

print("✅ Report successfully created and saved as 'student_report.html'")
