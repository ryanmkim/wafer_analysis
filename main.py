import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import zoom
from fpdf import FPDF
import os
from datetime import datetime

DATASET_PATH = 'LSWMD.pkl'
FIGURES_DIR = 'figures'
OUTPUT_DIR = 'output'

PATTERNS = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch']

PATTERN_DESCRIPTIONS = {
    'Center': 'Defects concentrated at the center of the wafer.',
    'Donut': 'Ring-shaped defect pattern with a clear center region.',
    'Edge-Loc': 'Defects localized to one section of the wafer edge.',
    'Edge-Ring': 'Defects forming a continuous ring around the wafer perimeter.',
    'Loc': 'Localized cluster of defects at an arbitrary position.',
    'Near-full': 'Defects covering most of the wafer surface.',
    'Random': 'Defects scattered randomly with no clear spatial pattern.',
    'Scratch': 'Linear streak of defects across the wafer surface.'
}

def load_data(filepath):
    df = pd.read_pickle(filepath)
    
    df['failureType'] = df['failureType'].apply(
        lambda x: x[0][0] if isinstance(x, np.ndarray) and len(x) > 0 and len(x[0]) > 0 else 'unknown'
    )
    
    df = df[df['failureType'].isin(PATTERNS)].copy()
    df = df.reset_index(drop=True)
    
    return df[['waferMap', 'failureType']]

def get_wafers_for_pattern(df, pattern):
    pattern_df = df[df['failureType'] == pattern]
    return pattern_df['waferMap'].tolist()

def calc_defect_density(wafer_map):
    good_die = np.sum(wafer_map == 1)
    defect_die = np.sum(wafer_map == 2)
    total_die = good_die + defect_die
    
    if total_die == 0:
        return 0.0
    
    return (defect_die / total_die) * 100

def calc_pattern_stats(df, pattern):
    wafer_maps = get_wafers_for_pattern(df, pattern)
    count = len(wafer_maps)
    
    densities = [calc_defect_density(w) for w in wafer_maps]
    
    return {
        'count': count,
        'percentage': (count / len(df)) * 100,
        'avg_density': np.mean(densities) if densities else 0,
        'std_density': np.std(densities) if densities else 0,
        'min_density': np.min(densities) if densities else 0,
        'max_density': np.max(densities) if densities else 0
    }

def calc_average_wafer(wafer_maps, target_size=50):
    if not wafer_maps:
        return np.zeros((target_size, target_size))
    
    resized_maps = []
    
    for wm in wafer_maps:
        binary_map = (wm == 2).astype(float)
        zoom_y = target_size / binary_map.shape[0]
        zoom_x = target_size / binary_map.shape[1]
        resized = zoom(binary_map, (zoom_y, zoom_x), order=1)
        resized = resized[:target_size, :target_size]
        if resized.shape != (target_size, target_size):
            padded = np.zeros((target_size, target_size))
            padded[:resized.shape[0], :resized.shape[1]] = resized
            resized = padded
        resized_maps.append(resized)
    
    stacked = np.stack(resized_maps, axis=0)
    return np.mean(stacked, axis=0)

def create_circular_mask(shape):
    h, w = shape
    center_y, center_x = h // 2, w // 2
    radius = min(center_y, center_x)
    y, x = np.ogrid[:h, :w]
    return ((x - center_x)**2 + (y - center_y)**2) <= radius**2

def plot_example_wafers(wafer_maps, pattern, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()
    
    n_samples = min(4, len(wafer_maps))
    indices = np.random.choice(len(wafer_maps), size=n_samples, replace=False)
    
    colors = ['white', '#3498db', '#e74c3c']
    cmap = ListedColormap(colors)
    
    for i, ax in enumerate(axes):
        if i < n_samples:
            wm = wafer_maps[indices[i]]
            display = wm.copy().astype(float)
            mask = create_circular_mask(wm.shape)
            display[~mask] = np.nan
            ax.imshow(display, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
            ax.set_title(f'Example {i+1}', fontsize=10)
        ax.axis('off')
    
    fig.suptitle(f'{pattern} Pattern Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_average_heatmap(avg_wafer, pattern, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    display = avg_wafer.copy()
    mask = create_circular_mask(avg_wafer.shape)
    display[~mask] = np.nan
    
    im = ax.imshow(display, cmap='YlOrRd', vmin=0, interpolation='bilinear')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Defect Probability', fontsize=10)
    
    ax.set_title(f'{pattern} - Average Defect Signature', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_density_histogram(densities, pattern, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.hist(densities, bins=30, color='#3498db', edgecolor='white', alpha=0.7)
    mean_density = np.mean(densities)
    ax.axvline(mean_density, color='#e74c3c', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_density:.2f}%')
    
    ax.set_xlabel('Defect Density (%)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'{pattern} - Defect Density Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_pattern_distribution(df, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    counts = df['failureType'].value_counts()
    pattern_counts = [counts.get(p, 0) for p in PATTERNS]
    
    bars = ax.bar(PATTERNS, pattern_counts, color='#3498db', edgecolor='white')
    
    for bar, count in zip(bars, pattern_counts):
        height = bar.get_height()
        ax.annotate(f'{count:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Defect Pattern', fontsize=11)
    ax.set_ylabel('Number of Wafers', fontsize=11)
    ax.set_title('Distribution of Defect Patterns in WM-811K Dataset', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

class WaferReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'B', 10)
            self.cell(0, 10, 'WM-811K Wafer Defect Field Guide', align='C', ln=True)
            self.ln(2)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def generate_pdf(all_stats, total_wafers, output_path):
    pdf = WaferReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 28)
    pdf.ln(80)
    pdf.cell(0, 15, 'WM-811K Wafer Defect Pattern', align='C', ln=True)
    pdf.cell(0, 15, 'Field Guide', align='C', ln=True)
    pdf.ln(30)
    pdf.set_font('Helvetica', '', 14)
    pdf.cell(0, 10, f'Total Wafers Analyzed: {total_wafers:,}', align='C', ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, datetime.now().strftime("%B %d, %Y"), align='C', ln=True)
    
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 15, 'Dataset Overview', ln=True)
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 11)
    pdf.multi_cell(0, 7, f'This report analyzes {total_wafers:,} labeled wafer maps from the WM-811K dataset, categorizing them into 8 distinct defect patterns.')
    pdf.ln(10)
    pdf.image(f'{FIGURES_DIR}/pattern_distribution.png', x=15, w=180)
    
    for pattern in PATTERNS:
        stats = all_stats[pattern]
        filename_base = pattern.lower().replace('-', '_')
        
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 22)
        pdf.cell(0, 15, pattern, ln=True)
        
        pdf.set_font('Helvetica', 'I', 11)
        pdf.multi_cell(0, 7, PATTERN_DESCRIPTIONS[pattern])
        pdf.ln(8)
        
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Statistics', ln=True)
        pdf.ln(2)
        
        pdf.set_font('Helvetica', '', 10)
        stats_data = [
            ('Count', f"{stats['count']:,} wafers"),
            ('Percentage of Dataset', f"{stats['percentage']:.2f}%"),
            ('Average Defect Density', f"{stats['avg_density']:.2f}%"),
            ('Std. Deviation', f"{stats['std_density']:.2f}%"),
            ('Min Density', f"{stats['min_density']:.2f}%"),
            ('Max Density', f"{stats['max_density']:.2f}%"),
        ]
        
        for label, value in stats_data:
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(80, 8, label, border=1, fill=True)
            pdf.cell(60, 8, value, border=1, ln=True)
        
        pdf.ln(8)
        
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Example Wafer Maps', ln=True)
        pdf.image(f'{FIGURES_DIR}/{filename_base}_examples.png', x=30, w=150)
        
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, f'{pattern} - Average Defect Signature', ln=True)
        pdf.ln(2)
        pdf.image(f'{FIGURES_DIR}/{filename_base}_average.png', x=50, w=110)
        
        pdf.ln(5)
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, f'{pattern} - Defect Density Distribution', ln=True)
        pdf.ln(2)
        pdf.image(f'{FIGURES_DIR}/{filename_base}_histogram.png', x=25, w=160)
    
    pdf.output(output_path)

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading dataset...")
    df = load_data(DATASET_PATH)
    total_wafers = len(df)
    print(f"Loaded {total_wafers:,} labeled wafers")
    
    print("Generating overview plot...")
    plot_pattern_distribution(df, f'{FIGURES_DIR}/pattern_distribution.png')
    
    all_stats = {}
    
    for pattern in PATTERNS:
        print(f"Processing {pattern}...")
        
        wafer_maps = get_wafers_for_pattern(df, pattern)
        stats = calc_pattern_stats(df, pattern)
        all_stats[pattern] = stats
        
        filename_base = pattern.lower().replace('-', '_')
        
        plot_example_wafers(
            wafer_maps, 
            pattern, 
            f'{FIGURES_DIR}/{filename_base}_examples.png'
        )
        
        avg_wafer = calc_average_wafer(wafer_maps)
        plot_average_heatmap(
            avg_wafer, 
            pattern, 
            f'{FIGURES_DIR}/{filename_base}_average.png'
        )
        
        densities = [calc_defect_density(w) for w in wafer_maps]
        plot_density_histogram(
            densities, 
            pattern, 
            f'{FIGURES_DIR}/{filename_base}_histogram.png'
        )
    
    print("Generating PDF...")
    pdf_path = f'{OUTPUT_DIR}/field_guide.pdf'
    generate_pdf(all_stats, total_wafers, pdf_path)
    
    print(f"\nDone! PDF saved to {pdf_path}")

if __name__ == '__main__':
    main()
