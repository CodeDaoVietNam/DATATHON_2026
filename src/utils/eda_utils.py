"""
eda_utils.py — Professional EDA Utility Functions
===================================================
Modular helper functions for the Datathon 2026 EDA notebooks.
Inspired by best-practice EDA notebooks with GridSpec layouts,
custom spine formatting, and annotated bar charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from IPython.display import display, HTML

# ─── GLOBAL STYLE SETTINGS ───────────────────────────────────────────────────
PALETTE = 'magma'
TITLE_COLOR = '#2c3e50'
LABEL_COLOR = 'dimgrey'
BG_COLOR = '#f8f9fa'
HEADER_BG = '#34495e'
HEADER_COLOR = 'white'
MISSING_HEADER_BG = '#e74c3c'

# ─── SPINE / AXIS FORMATTING ─────────────────────────────────────────────────

def format_spines(ax, right_border=False):
    """
    Remove top, right (optionally) and style bottom/left spines.
    Mimics the reference notebook's clean chart look.
    """
    ax.spines['top'].set_visible(False)
    if not right_border:
        ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('lightgrey')
    ax.spines['bottom'].set_color('lightgrey')
    ax.tick_params(colors='grey', which='both')
    ax.yaxis.label.set_color(LABEL_COLOR)
    ax.xaxis.label.set_color(LABEL_COLOR)
    ax.title.set_color(TITLE_COLOR)


def set_plot_style():
    """Apply a consistent, professional matplotlib/seaborn style."""
    sns.set_style('white')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
    })


# ─── ANNOTATE BARS ───────────────────────────────────────────────────────────

class AnnotateBars:
    """
    Annotates bar/barh charts with value labels.
    Inspired by the reference notebook's AnnotateBars class.
    """
    def __init__(self, n_dec=0, font_size=10, color='black', prefix='', suffix=''):
        self.n_dec = n_dec
        self.font_size = font_size
        self.color = color
        self.prefix = prefix
        self.suffix = suffix

    def horizontal(self, ax, negative_change=False):
        for p in ax.patches:
            w = p.get_width()
            if w == 0:
                continue
            formatting = f'{{:{"."+str(self.n_dec)+"f" if self.n_dec > 0 else ".0f"}}}'
            text = self.prefix + formatting.format(w) + self.suffix
            x = w * 1.01 if not negative_change else w - abs(w) * 0.1
            y = p.get_y() + p.get_height() / 2
            ax.annotate(text, (x, y), fontsize=self.font_size, color=self.color,
                        va='center', ha='left')

    def vertical(self, ax, negative_change=False):
        for p in ax.patches:
            h = p.get_height()
            if h == 0:
                continue
            formatting = f'{{:{"."+str(self.n_dec)+"f" if self.n_dec > 0 else ".0f"}}}'
            text = self.prefix + formatting.format(h) + self.suffix
            x = p.get_x() + p.get_width() / 2
            y = h * 1.01 if not negative_change else h - abs(h) * 0.05
            ax.annotate(text, (x, y), fontsize=self.font_size, color=self.color,
                        va='bottom', ha='center')


# ─── SINGLE COUNT / BAR PLOTS ────────────────────────────────────────────────

def single_countplot(x=None, y=None, hue=None, ax=None, df=None, palette='viridis', order=None):
    """
    Seaborn countplot wrapper with value annotations.
    """
    if order is None:
        if y is not None:
            order = df[y].value_counts().index
        elif x is not None:
            order = df[x].value_counts().index
    if y is not None:
        sns.countplot(y=y, hue=hue, data=df, ax=ax, palette=palette, order=order)
        AnnotateBars(n_dec=0, font_size=9, color='black').horizontal(ax)
    else:
        sns.countplot(x=x, hue=hue, data=df, ax=ax, palette=palette, order=order)
        AnnotateBars(n_dec=0, font_size=9, color='black').vertical(ax)
    format_spines(ax, right_border=False)


# ─── DATA OVERVIEW TABLE ─────────────────────────────────────────────────────

def data_overview(dfs_dict):
    """
    Build a professional summary DataFrame from a dict of DataFrames.
    Shows rows, columns, missing values, % missing, duplicates.
    """
    records = []
    for name, df in dfs_dict.items():
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isnull().sum().sum()
        records.append({
            'Dataset': name.upper(),
            'Rows': len(df),
            'Columns': df.shape[1],
            'Missing Values': int(total_missing),
            '% Missing': float(total_missing / total_cells * 100) if total_cells else 0,
            'Duplicates': int(df.duplicated().sum()),
        })
    df_overview = pd.DataFrame(records)

    styled = (df_overview.style
              .background_gradient(subset=['Rows'], cmap='Blues')
              .background_gradient(subset=['Missing Values', '% Missing'], cmap='Reds')
              .format({
                  'Rows': '{:,}', 'Columns': '{:,}',
                  'Missing Values': '{:,}', '% Missing': '{:.2f}%',
                  'Duplicates': '{:,}'
              })
              .set_properties(**{'text-align': 'center'})
              .set_table_styles([{
                  'selector': 'th',
                  'props': [('background-color', HEADER_BG),
                            ('color', HEADER_COLOR),
                            ('font-size', '14px'),
                            ('text-align', 'center')]
              }])
              .hide(axis='index'))
    return styled


# ─── INSPECT TABLE ────────────────────────────────────────────────────────────

def inspect_table(name, df, n_rows=3):
    """Display a table's schema and first n rows with professional styling."""
    html = f"""
    <div style="background-color:{BG_COLOR}; padding:15px;
                border-left: 5px solid #2980b9; margin-bottom:15px;">
        <h3 style="margin:0; color:{TITLE_COLOR};">
            {name.upper()} | {len(df):,} rows × {df.shape[1]} cols
        </h3>
    </div>
    """
    display(HTML(html))

    # DataType info
    dtypes_df = (pd.DataFrame(df.dtypes, columns=['DataType'])
                 .reset_index().rename(columns={'index': 'Feature'}))
    display(dtypes_df.style
            .set_table_styles([{'selector': 'th',
                                'props': [('background-color', '#ecf0f1'),
                                          ('color', TITLE_COLOR)]}])
            .hide(axis='index'))

    # First rows
    print(f'\nFirst {n_rows} rows:')
    display(df.head(n_rows).style
            .set_table_styles([{'selector': 'th',
                                'props': [('background-color', '#ecf0f1'),
                                          ('color', TITLE_COLOR)]}])
            .hide(axis='index'))


# ─── MISSING VALUE ANALYSIS ──────────────────────────────────────────────────

def analyze_missing_values(dfs_dict):
    """Analyze and display missing values with the exact requested format."""
    records = []
    for name, df in dfs_dict.items():
        n = len(df)
        for col in df.columns:
            qtd_null = df[col].isnull().sum()
            percent_null = qtd_null / n if n > 0 else 0
            
            # Count unique categorical values if object type
            qtd_cat = 0
            if df[col].dtype == 'object':
                qtd_cat = df[col].nunique()
                
            records.append({
                'dataset_name': name,
                'feature': col,
                'qtd_null': int(qtd_null),
                'percent_null': float(percent_null),
                'dtype': str(df[col].dtype),
                'qtd_cat': int(qtd_cat)
            })
            
    df_missing = pd.DataFrame(records)
    # Highlight nulls
    styled = (df_missing.style
              .background_gradient(subset=['percent_null'], cmap='OrRd')
              .format({'percent_null': '{:.6f}'}))
    display(styled)


# ─── MISSING VALUE HEATMAP ───────────────────────────────────────────────────

def plot_missing_heatmap(dfs_dict, figsize=(16, 5)):
    """
    Plot a heatmap of missing values for each dataset (reference-style).
    Only shows datasets that actually have missing values.
    """
    datasets_with_missing = {k: v for k, v in dfs_dict.items()
                             if v.isnull().sum().sum() > 0}
    if not datasets_with_missing:
        display(HTML("<h4 style='color:green;'>No missing values to visualize!</h4>"))
        return

    n = len(datasets_with_missing)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0], figsize[1]))
    if n == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, datasets_with_missing.items()):
        # Only show columns with missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            sns.heatmap(df[missing_cols].isnull().astype(int),
                        cbar=False, ax=ax, cmap='magma_r',
                        yticklabels=False)
            ax.set_title(f'{name.upper()}\n(Missing Pattern)', fontsize=11, color=TITLE_COLOR)
            ax.tick_params(axis='x', rotation=45)
        format_spines(ax)
    plt.suptitle('Missing Value Heatmap', fontsize=14, color=TITLE_COLOR, y=1.02)
    plt.tight_layout()
    plt.show()


# ─── DISTRIBUTION PLOTS (NUMERIC) ────────────────────────────────────────────

def plot_numeric_distributions(df, cols=None, n_cols=3, figsize_per_plot=(5, 3.5)):
    """
    Plot histograms + KDE for numeric columns in a professional GridSpec layout.
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n = len(cols)
    if n == 0:
        return
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize_per_plot[0] * n_cols,
                                      figsize_per_plot[1] * n_rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(cols):
        ax = axes[i]
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color='#3498db', edgecolor='white')
        ax.set_title(col, fontsize=11, color=TITLE_COLOR)
        ax.set_xlabel('')
        format_spines(ax)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Numeric Feature Distributions', fontsize=14, color=TITLE_COLOR, y=1.01)
    plt.tight_layout()
    plt.show()


# ─── CATEGORICAL DISTRIBUTION (PIE + BAR) ────────────────────────────────────

def plot_categorical_distributions(df, cols=None, top_n=10, figsize_per_plot=(6, 4)):
    """
    Plot horizontal bar charts for categorical columns.
    """
    if cols is None:
        cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    n = len(cols)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(figsize_per_plot[0], figsize_per_plot[1] * n))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        vc = df[col].value_counts().head(top_n)
        sns.barplot(x=vc.values, y=vc.index, ax=ax, palette='viridis')
        ax.set_title(f'{col} (Top {min(top_n, len(vc))})', fontsize=11, color=TITLE_COLOR)
        ax.set_xlabel('Count')
        ax.set_ylabel('')
        AnnotateBars(n_dec=0, font_size=9, color='black').horizontal(ax)
        format_spines(ax)

    plt.suptitle('Categorical Feature Distributions', fontsize=14, color=TITLE_COLOR, y=1.01)
    plt.tight_layout()
    plt.show()


# ─── TIME-SERIES DASHBOARD ───────────────────────────────────────────────────

def plot_timeseries_dashboard(df_sales, date_col='Date', revenue_col='Revenue',
                               cogs_col='COGS', figsize=(16, 12)):
    """
    Professional 2x2 GridSpec time-series dashboard:
      - Top-left:  Revenue trend (line)
      - Top-right: COGS trend (line)
      - Bottom-left:  Gross Profit trend (line)
      - Bottom-right: Gross Margin % (line + fill)
    """
    from matplotlib.gridspec import GridSpec

    df = df_sales.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Compute metrics
    df['Gross_Profit'] = df[revenue_col] - df[cogs_col]
    df['Gross_Margin_%'] = (df['Gross_Profit'] / df[revenue_col]) * 100

    # Monthly aggregation
    df['YearMonth'] = df[date_col].dt.to_period('M')
    monthly = df.groupby('YearMonth').agg({
        revenue_col: 'sum',
        cogs_col: 'sum',
        'Gross_Profit': 'sum',
    }).reset_index()
    monthly['Gross_Margin_%'] = (monthly['Gross_Profit'] / monthly[revenue_col]) * 100
    monthly['YearMonth'] = monthly['YearMonth'].astype(str)

    set_plot_style()
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = GridSpec(2, 2, figure=fig)

    # --- Revenue ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(monthly['YearMonth'], monthly[revenue_col] / 1e6, color='#2ecc71', linewidth=2, marker='o', markersize=3)
    ax1.fill_between(range(len(monthly)), monthly[revenue_col].values / 1e6, alpha=0.15, color='#2ecc71')
    ax1.set_title('Monthly Revenue (M VND)', fontsize=12, color=TITLE_COLOR)
    ax1.set_ylabel('Revenue (Millions)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    format_spines(ax1)

    # --- COGS ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(monthly['YearMonth'], monthly[cogs_col] / 1e6, color='#e74c3c', linewidth=2, marker='o', markersize=3)
    ax2.fill_between(range(len(monthly)), monthly[cogs_col].values / 1e6, alpha=0.15, color='#e74c3c')
    ax2.set_title('Monthly COGS (M VND)', fontsize=12, color=TITLE_COLOR)
    ax2.set_ylabel('COGS (Millions)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    format_spines(ax2)

    # --- Gross Profit ---
    ax3 = fig.add_subplot(gs[1, 0])
    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in monthly['Gross_Profit'].values]
    ax3.bar(range(len(monthly)), monthly['Gross_Profit'].values / 1e6, color=colors, alpha=0.8)
    ax3.set_title('Monthly Gross Profit (M VND)', fontsize=12, color=TITLE_COLOR)
    ax3.set_ylabel('Gross Profit (Millions)')
    ax3.set_xticks(range(0, len(monthly), max(1, len(monthly)//12)))
    ax3.set_xticklabels(monthly['YearMonth'].values[::max(1, len(monthly)//12)], rotation=45, ha='right')
    format_spines(ax3)

    # --- Gross Margin % ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(monthly['YearMonth'], monthly['Gross_Margin_%'], color='#3498db', linewidth=2, marker='s', markersize=3)
    ax4.fill_between(range(len(monthly)), monthly['Gross_Margin_%'].values, alpha=0.15, color='#3498db')
    ax4.axhline(y=monthly['Gross_Margin_%'].mean(), color='orange', linestyle='--', linewidth=1, label=f"Avg: {monthly['Gross_Margin_%'].mean():.1f}%")
    ax4.set_title('Monthly Gross Margin %', fontsize=12, color=TITLE_COLOR)
    ax4.set_ylabel('Margin (%)')
    ax4.legend(fontsize=9)
    ax4.tick_params(axis='x', rotation=45)
    ax4.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    format_spines(ax4)

    plt.suptitle('Financial Time-Series Dashboard', fontsize=16, color=TITLE_COLOR, fontweight='bold', y=1.02)
    plt.show()


# ─── CORRELATION HEATMAP ─────────────────────────────────────────────────────

def plot_correlation_heatmap(df, figsize=(10, 8), title='Feature Correlation Heatmap'):
    """Plot a triangular correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title(title, fontsize=14, color=TITLE_COLOR, pad=15)
    plt.tight_layout()
    plt.show()


# ─── SECTION HEADER HELPER ───────────────────────────────────────────────────

def section_header(title, subtitle=''):
    """Display a professional section header in the notebook."""
    sub = f"<p style='margin:5px 0 0 0; color:#7f8c8d; font-size:13px;'>{subtitle}</p>" if subtitle else ''
    html = f"""
    <div style="background: linear-gradient(135deg, #2c3e50, #3498db);
                padding: 20px; border-radius: 8px; margin: 20px 0;">
        <h2 style="margin:0; color:white; font-size:22px;">
            {title}
        </h2>
        {sub}
    </div>
    """
    display(HTML(html))


# ─── SUMMARY KPI CARDS ───────────────────────────────────────────────────────

def display_kpi_cards(kpis):
    """
    Display a row of KPI cards.
    kpis: list of dicts with keys 'label', 'value', 'color' (optional)
    """
    cards_html = '<div style="display:flex; gap:15px; flex-wrap:wrap; margin:15px 0;">'
    for kpi in kpis:
        color = kpi.get('color', '#3498db')
        cards_html += f"""
        <div style="flex:1; min-width:180px; background:white; border-left:5px solid {color};
                    padding:15px; border-radius:5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <p style="margin:0; font-size:12px; color:#7f8c8d; text-transform:uppercase;">{kpi['label']}</p>
            <p style="margin:5px 0 0 0; font-size:24px; font-weight:bold; color:{TITLE_COLOR};">{kpi['value']}</p>
        </div>
        """
    cards_html += '</div>'
    display(HTML(cards_html))
