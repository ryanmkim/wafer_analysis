import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import zoom
from fpdf import FPDF
import os
from datetime import datetime

DATASET = 'LSWMD.pkl'
FIG_DIR = 'figures'
OUT_DIR = 'output'

PATTERNS = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch']

PATTERN_DESC = {
    'Center':    'Defects concentrated at the center of the wafer.',
    'Donut':     'Ring-shaped defect pattern with a clear center region.',
    'Edge-Loc':  'Defects localized to one section of the wafer edge.',
    'Edge-Ring': 'Defects forming a continuous ring around the wafer perimeter.',
    'Loc':       'Localized cluster of defects at an arbitrary position.',
    'Near-full': 'Defects covering most of the wafer surface.',
    'Random':    'Defects scattered randomly with no clear spatial pattern.',
    'Scratch':   'Linear streak of defects across the wafer surface.',
}


def load_data(path):
    df = pd.read_pickle(path)
    df['failureType'] = df['failureType'].apply(
        lambda x: x[0][0] if isinstance(x, np.ndarray) and len(x) > 0 and len(x[0]) > 0 else 'unknown'
    )
    df = df[df['failureType'].isin(PATTERNS)].copy().reset_index(drop=True)
    return df[['waferMap', 'failureType']]


def wafers_for(df, pattern):
    return df.loc[df['failureType'] == pattern, 'waferMap'].tolist()


def defect_density(wm):
    good = np.sum(wm == 1)
    bad = np.sum(wm == 2)
    total = good + bad
    return (bad / total) * 100 if total else 0.0


def pattern_stats(df, pattern):
    maps = wafers_for(df, pattern)
    n = len(maps)
    densities = [defect_density(w) for w in maps]
    return {
        'count':       n,
        'pct':         (n / len(df)) * 100,
        'avg_density': np.mean(densities) if densities else 0,
        'std_density': np.std(densities) if densities else 0,
        'min_density': np.min(densities) if densities else 0,
        'max_density': np.max(densities) if densities else 0,
    }


def avg_wafer(maps, size=50):
    if not maps:
        return np.zeros((size, size))

    resized = []
    for wm in maps:
        binary = (wm == 2).astype(float)
        zy, zx = size / binary.shape[0], size / binary.shape[1]
        r = zoom(binary, (zy, zx), order=1)[:size, :size]
        if r.shape != (size, size):
            pad = np.zeros((size, size))
            pad[:r.shape[0], :r.shape[1]] = r
            r = pad
        resized.append(r)

    return np.mean(np.stack(resized), axis=0)


def circ_mask(shape):
    h, w = shape
    cy, cx = h // 2, w // 2
    radius = min(cy, cx)
    y, x = np.ogrid[:h, :w]
    return ((x - cx)**2 + (y - cy)**2) <= radius**2


def plot_examples(maps, pattern, path):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    n = min(4, len(maps))
    picks = np.random.choice(len(maps), size=n, replace=False)
    cmap = ListedColormap(['white', '#3498db', '#e74c3c'])

    for i, ax in enumerate(axes):
        if i < n:
            wm = maps[picks[i]].copy().astype(float)
            wm[~circ_mask(wm.shape)] = np.nan
            ax.imshow(wm, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
            ax.set_title(f'Example {i+1}', fontsize=10)
        ax.axis('off')

    fig.suptitle(f'{pattern} Pattern Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_heatmap(avg, pattern, path):
    fig, ax = plt.subplots(figsize=(6, 5))

    disp = avg.copy()
    disp[~circ_mask(avg.shape)] = np.nan

    im = ax.imshow(disp, cmap='YlOrRd', vmin=0, interpolation='bilinear')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Defect Probability', fontsize=10)
    ax.set_title(f'{pattern} - Average Defect Signature', fontsize=12, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_histogram(densities, pattern, path):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(densities, bins=30, color='#3498db', edgecolor='white', alpha=0.7)
    mu = np.mean(densities)
    ax.axvline(mu, color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {mu:.2f}%')
    ax.set_xlabel('Defect Density (%)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'{pattern} - Defect Density Distribution', fontsize=12, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_distribution(df, path):
    fig, ax = plt.subplots(figsize=(10, 5))

    counts = df['failureType'].value_counts()
    vals = [counts.get(p, 0) for p in PATTERNS]
    bars = ax.bar(PATTERNS, vals, color='#3498db', edgecolor='white')

    for bar, c in zip(bars, vals):
        ax.annotate(f'{c:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Defect Pattern', fontsize=11)
    ax.set_ylabel('Number of Wafers', fontsize=11)
    ax.set_title('Distribution of Defect Patterns in WM-811K Dataset', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
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


def build_pdf(all_stats, n_wafers, out_path):
    pdf = WaferReport()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 28)
    pdf.ln(80)
    pdf.cell(0, 15, 'WM-811K Wafer Defect Pattern', align='C', ln=True)
    pdf.cell(0, 15, 'Field Guide', align='C', ln=True)
    pdf.ln(30)
    pdf.set_font('Helvetica', '', 14)
    pdf.cell(0, 10, f'Total Wafers Analyzed: {n_wafers:,}', align='C', ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, datetime.now().strftime('%B %d, %Y'), align='C', ln=True)

    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 15, 'Dataset Overview', ln=True)
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 11)
    pdf.multi_cell(0, 7,
        f'This report analyzes {n_wafers:,} labeled wafer maps from the '
        f'WM-811K dataset, categorizing them into 8 distinct defect patterns.')
    pdf.ln(10)
    pdf.image(f'{FIG_DIR}/pattern_distribution.png', x=15, w=180)

    for pat in PATTERNS:
        st = all_stats[pat]
        slug = pat.lower().replace('-', '_')

        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 22)
        pdf.cell(0, 15, pat, ln=True)

        pdf.set_font('Helvetica', 'I', 11)
        pdf.multi_cell(0, 7, PATTERN_DESC[pat])
        pdf.ln(8)

        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Statistics', ln=True)
        pdf.ln(2)

        pdf.set_font('Helvetica', '', 10)
        rows = [
            ('Count',                  f"{st['count']:,} wafers"),
            ('Percentage of Dataset',  f"{st['pct']:.2f}%"),
            ('Average Defect Density', f"{st['avg_density']:.2f}%"),
            ('Std. Deviation',         f"{st['std_density']:.2f}%"),
            ('Min Density',            f"{st['min_density']:.2f}%"),
            ('Max Density',            f"{st['max_density']:.2f}%"),
        ]
        for label, val in rows:
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(80, 8, label, border=1, fill=True)
            pdf.cell(60, 8, val, border=1, ln=True)

        pdf.ln(8)
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Example Wafer Maps', ln=True)
        pdf.image(f'{FIG_DIR}/{slug}_examples.png', x=30, w=150)

        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, f'{pat} - Average Defect Signature', ln=True)
        pdf.ln(2)
        pdf.image(f'{FIG_DIR}/{slug}_average.png', x=50, w=110)

        pdf.ln(5)
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, f'{pat} - Defect Density Distribution', ln=True)
        pdf.ln(2)
        pdf.image(f'{FIG_DIR}/{slug}_histogram.png', x=25, w=160)

    pdf.output(out_path) 


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)


    df = load_data(DATASET)
    n = len(df)
    print(f'got {n:,} labeled wafers')

    print('overview plot...')
    plot_distribution(df, f'{FIG_DIR}/pattern_distribution.png')

    all_stats = {}

    for pat in PATTERNS:
        print(f'  {pat}...')

        maps = wafers_for(df, pat)
        all_stats[pat] = pattern_stats(df, pat)
        slug = pat.lower().replace('-', '_')

        plot_examples(maps, pat, f'{FIG_DIR}/{slug}_examples.png')

        avg = avg_wafer(maps)
        plot_heatmap(avg, pat, f'{FIG_DIR}/{slug}_average.png')

        dens = [defect_density(w) for w in maps]
        plot_histogram(dens, pat, f'{FIG_DIR}/{slug}_histogram.png')

    print('building pdf...')
    pdf_path = f'{OUT_DIR}/field_guide.pdf'
    build_pdf(all_stats, n, pdf_path)

    print(f'done: {pdf_path}')


if __name__ == '__main__':
    main()
