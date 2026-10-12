import matplotlib.pyplot as plt
import numpy as np

# Set font and style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Colors
teal = '#2A9D8F'
coral = '#E76F51'
navy = '#264653'
yellow = '#E9C46A'

# -------------------------------------------------------------
# Graph 1: Resolution & Aspect Ratio Distribution (DIV2K + Flickr2K + OTS)
# -------------------------------------------------------------
resolutions = ['2K Landscape\n(2040x1356)', '2K Portrait\n(1356x2040)', 'Square\n(~1500x1500)', 'FHD / Lower\n(<=1080p)']
counts = [2450, 1820, 780, 461]
colors_res = [navy, teal, yellow, coral]

bars = axs[0].bar(resolutions, counts, color=colors_res, width=0.55, edgecolor='black', linewidth=0.8)
axs[0].set_title('(A) HR Dataset Resolution Distribution', fontsize=12, fontweight='bold', pad=15)
axs[0].set_ylabel('Number of Images', fontsize=11)
axs[0].set_ylim(0, 3000)

for bar in bars:
    yval = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width()/2.0, yval + 50, f'{yval:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

axs[0].text(0.5, -0.22, "Analysis: Over 75% of training samples are 2K high-resolution images (DIV2K & Flickr2K),\nproviding rich high-frequency textures for synthetic degradation.", 
            ha='center', transform=axs[0].transAxes, fontsize=9.5, style='italic', color='#334155')

# -------------------------------------------------------------
# Graph 2: Simulated Degradation Parameters Range in Training
# -------------------------------------------------------------
degradation_types = [
    'Gaussian Blur Kernel\n(Kernel size: 7 - 21)',
    'Gaussian Noise Level\n(Sigma: 1 - 30)',
    'Poisson Noise Scale\n(Scale: 0.05 - 3.0)',
    'JPEG Quality Factor\n(Q: 30 - 95)',
    'Sinc Filter Applied\n(Probability: ~80%)'
]
coverage = [100, 90, 80, 95, 80]

y_pos = np.arange(len(degradation_types))
bars2 = axs[1].barh(y_pos, coverage, color=teal, height=0.55, edgecolor='black', linewidth=0.8)
axs[1].set_yticks(y_pos)
axs[1].set_yticklabels(degradation_types, fontsize=10)
axs[1].invert_yaxis()
axs[1].set_xlabel('Coverage in Training Batches (%)', fontsize=11)
axs[1].set_xlim(0, 115)
axs[1].set_title('(B) Degradation Parameter Coverage (EDA)', fontsize=12, fontweight='bold', pad=15)

for bar in bars2:
    xval = bar.get_width()
    axs[1].text(xval + 2, bar.get_y() + bar.get_height()/2.0, f'{xval}%', ha='left', va='center', fontsize=10, fontweight='bold')

axs[1].text(0.5, -0.22, "Analysis: Covers realistic ranges of noise, blur, and severe JPEG compression,\nenabling robust generalization on unseen real-world images (Blind SR).", 
            ha='center', transform=axs[1].transAxes, fontsize=9.5, style='italic', color='#334155')

plt.tight_layout()
output_img = 'eda_dataset_distribution.png'
plt.savefig(output_img, bbox_inches='tight')
print(f"EDA graph successfully saved to: {output_img}")
