from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import seaborn as sns
from IPython.display import display, HTML
from scipy.stats import pearsonr

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'dataset'
FIG_DIR = BASE_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True, parents=True)

PALETTE = {
    'primary': '#2D6A4F',
    'secondary': '#52B788',
    'accent': '#F4A261',
    'danger': '#E76F51',
    'neutral': '#6C757D',
    'light': '#F8F9FA',
    'dark': '#212529',
    'blue': '#457B9D',
    'purple': '#9B5DE5',
    'stockout': '#E63946',
    'overstock': '#F4A261',
    'stable': '#2A9D8F',
    'warning': '#E9C46A',
}

def _fmt_percent(val: float) -> str:
    """Format value as percentage string. Parameters:     val (float): Value between 0 and 1 (or 0-100 if using % directly)
    Returns:
        str: Formatted percentage string (e.g., '50.1%')
    """
    return f"{val * 100:.1f}%" if val < 1 else f"{val:.1f}%"


def _fmt_currency(val: float, decimals: int = 0) -> str:
    """Format value as Vietnamese currency string.
    
    Parameters:
        val (float): Numeric value to format
        decimals (int): Number of decimal places
    Returns:
        str: Formatted currency string (e.g., '1.50 Tỷ')
    """
    if val >= 1e9:
        return f"{val / 1e9:.{decimals}f} Tỷ"
    elif val >= 1e6:
        return f"{val / 1e6:.1f} Tr"
    return f"{val:,.0f}"


def _fmt_number(val: float, decimals: int = 0) -> str:
    """Format value as localized number string.
    
    Parameters:
        val (float): Numeric value to format
        decimals (int): Number of decimal places
    Returns:
        str: Formatted number string with thousands separator
    """
    return f"{val:,.{decimals}f}"


def _label_vbars(
    ax: plt.Axes, fmt_fn=None, offset: float = 0.02
) -> None:
    """Attach labels to vertical bar chart.
    
    Parameters:
        ax (plt.Axes): Matplotlib axes object
        fmt_fn (callable, optional): Function to format label values
        offset (float): Relative offset from bar top for label
    """
    if fmt_fn is None:
        fmt_fn = lambda v: f"{v:.1f}"
    
    for bar in ax.patches:
        yval = bar.get_height()
        offset_px = yval * offset if offset > 0 else abs(offset)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + offset_px,
            fmt_fn(yval),
            ha='center', va='bottom',
            fontsize=10, fontweight='bold',
            color='#2B3A42'
        )


def _label_hbars(
    ax: plt.Axes, fmt_fn=None, offset: float = 0.01
) -> None:
    """Attach labels to horizontal bar chart.
    
    Parameters:
        ax (plt.Axes): Matplotlib axes object
        fmt_fn (callable, optional): Function to format label values
        offset (float): Relative offset from bar end
    """
    if fmt_fn is None:
        fmt_fn = lambda v: f"{v:.1f}"
    
    max_x = ax.get_xlim()[1]
    for bar in ax.patches:
        xval = bar.get_width()
        offset_px = max_x * offset
        ax.text(
            xval + offset_px,
            bar.get_y() + bar.get_height() / 2,
            fmt_fn(xval),
            ha='left', va='center',
            fontsize=10, fontweight='bold',
            color='#2B3A42'
        )


def format_spines(ax: plt.Axes, right_border: bool = False) -> None:
    """Format chart spines for a clean presentation.
    
    Parameters:
        ax (plt.Axes): Matplotlib axes object
        right_border (bool): Whether to keep right spine visible (for twinx)
    """
    ax.spines['top'].set_visible(False)
    if not right_border:
        ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(colors='#5B6D74', which='both')
    ax.yaxis.label.set_color('#2B3A42')
    ax.xaxis.label.set_color('#2B3A42')
    ax.title.set_color('#2B3A42')


def set_plot_style() -> None:
    """Apply a consistent matplotlib/seaborn style for EDA charts."""
    sns.set_style('whitegrid')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#DDDDDD',
        'axes.titleweight': 'bold',
        'axes.titlecolor': '#2B3A42',
        'axes.labelcolor': '#2B3A42',
        'xtick.color': '#5B6D74',
        'ytick.color': '#5B6D74',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'grid.color': '#E5E5E5',
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'legend.frameon': False,
        'figure.dpi': 120,
    })


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns with date-like names to datetime.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
    Returns:
        pd.DataFrame: Dataframe with datetime columns converted
    """
    
    date_cols = [col for col in df.columns
                 if 'date' in col.lower() or col == 'Date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def load_datasets() -> Dict[str, pd.DataFrame]:
    """Load all dataset CSV files and parse datetime columns.
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of loaded DataFrames keyed by dataset name
    
    Raises:
        FileNotFoundError: If any required dataset file is missing
    """
    file_map = {
        'orders': 'orders.csv',
        'order_items': 'order_items.csv',
        'products': 'products.csv',
        'promotions': 'promotions.csv',
        'inventory': 'inventory.csv',
        'returns': 'returns.csv',
        'sales': 'sales.csv',
        'traffic': 'web_traffic.csv',
        'customers': 'customers.csv',
        'geography': 'geography.csv',
        'shipments': 'shipments.csv',
        'payments': 'payments.csv',
        'reviews': 'reviews.csv',
        'sample_submission': 'sample_submission.csv',
    }
    dfs = {}
    for key, rel_path in file_map.items():
        file_path = DATA_DIR / rel_path
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        df = pd.read_csv(file_path, low_memory=False)
        dfs[key] = _parse_dates(df)
    return dfs


def prepare_master_df(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Prepare a unified transactional master table for EDA.
    
    Parameters:
        dfs (Dict[str, pd.DataFrame]): Dictionary of loaded DataFrames
    Returns:
        pd.DataFrame: Merged master dataframe with features
    """
    orders = dfs['orders']
    items = dfs['order_items'].copy() 
    products = dfs['products']
    promotions = dfs['promotions']

    item_features = [
        'product_id', 'category', 'segment', 'size',
        'price', 'cogs', 'product_name'
    ]
    items = items.merge(products[item_features], on='product_id', how='left')
    items['net_revenue'] = (
        items['unit_price'] * items['quantity'] - items['discount_amount']
    )
    items['gross_profit'] = (
        items['net_revenue'] - items['cogs'] * items['quantity']
    )
    
    items['has_promo'] = items[
        ['promo_id', 'promo_id_2']
    ].notna().any(axis=1).astype(int)

    promo_cols = ['promo_id', 'promo_type', 'stackable_flag']
    items = items.merge(promotions[promo_cols], on='promo_id', how='left')
    master = items.merge(
        orders[[
            'order_id', 'order_date', 'customer_id', 'order_status',
            'payment_method', 'device_type', 'order_source', 'zip'
        ]],
        on='order_id',
        how='left'
    )
    
    master['order_year'] = master['order_date'].dt.year
    master['order_month'] = master['order_date'].dt.to_period('M').astype(str)
    master['order_day'] = master['order_date'].dt.date
    return master


def build_business_kpis(
    master_df: pd.DataFrame, dfs: Dict[str, pd.DataFrame]
) -> Dict[str, float]:
    """Compute high-level business KPI values for summary cards.
    
    Parameters:
        master_df (pd.DataFrame): Master transaction dataframe
        dfs (Dict[str, pd.DataFrame]): Dictionary of source datasets
    Returns:
        Dict[str, float]: Dictionary of computed KPI values
    """
    orders = dfs['orders']
    returns = dfs['returns']
    customers = dfs['customers']
    reviews = dfs['reviews']
    inventory = dfs['inventory']

    total_revenue = master_df['net_revenue'].sum()
    total_cogs = (master_df['cogs'] * master_df['quantity']).sum()
    
    gross_margin = (
        ((total_revenue - total_cogs) / total_revenue * 100)
        if total_revenue else 0.0
    )
    total_orders = orders['order_id'].nunique()
    cancel_rate = (orders['order_status'] == 'cancelled').mean() * 100
    return_rate_order = (
        (returns['order_id'].nunique() / total_orders * 100)
        if total_orders else 0.0
    )
    return_rate_item = (
        (returns['return_quantity'].sum() / master_df['quantity'].sum() * 100)
        if master_df['quantity'].sum() else 0.0
    )
    total_customers = customers['customer_id'].nunique()
    avg_rating = (reviews['rating'].mean()
                  if len(reviews) else np.nan)
    low_rating_rate = (
        (reviews['rating'] <= 2).mean() * 100 if len(reviews) else 0.0
    )
    total_refund = returns['refund_amount'].sum()
    promo_penetration = master_df['has_promo'].mean() * 100
    avg_order_value = (
        master_df.groupby('order_id')['net_revenue'].sum().mean()
    )
    stockout_rate = (
        (inventory['stockout_flag'].mean() * 100)
        if len(inventory) else 0.0
    )
    overstock_rate = (
        (inventory['overstock_flag'].mean() * 100)
        if len(inventory) else 0.0
    )

    kpis = {
        'total_revenue': total_revenue,
        'total_cogs': total_cogs,
        'gross_margin': gross_margin,
        'total_orders': total_orders,
        'cancel_rate': cancel_rate,
        'return_rate_order': return_rate_order,
        'return_rate_item': return_rate_item,
        'total_customers': total_customers,
        'avg_rating': avg_rating,
        'low_rating_rate': low_rating_rate,
        'total_refund': total_refund,
        'promo_penetration': promo_penetration,
        'avg_order_value': avg_order_value,
        'stockout_rate': stockout_rate,
        'overstock_rate': overstock_rate,
    }
    return kpis


def build_kpi_cards(kpis: Dict[str, float]) -> list:
    """Build card definitions from the business KPI dictionary.
    
    Parameters:
        kpis (Dict[str, float]): Dictionary of KPI values
    Returns:
        list: List of card definition dictionaries
    """
    # [OPTIMIZED] Used list of tuples for more concise definition
    card_specs = [
        ('Tổng Doanh Thu', f"{kpis['total_revenue'] / 1e9:.2f} Tỷ",
         PALETTE['primary']),
        ('Gross Margin', f"{kpis['gross_margin']:.1f}%",
         PALETTE['secondary']),
        ('Tỷ lệ hủy đơn', f"{kpis['cancel_rate']:.1f}%",
         PALETTE['danger']),
        ('Tỷ lệ hoàn đơn', f"{kpis['return_rate_order']:.1f}%",
         PALETTE['accent']),
        ('Tỷ lệ promo', f"{kpis['promo_penetration']:.1f}%",
         PALETTE['blue']),
        ('Stockout rate', f"{kpis['stockout_rate']:.1f}%",
         PALETTE['stockout']),
        ('Overstock rate', f"{kpis['overstock_rate']:.1f}%",
         PALETTE['overstock']),
        ('Rating trung bình', f"{kpis['avg_rating']:.1f}",
         PALETTE['accent']),
        ('AOV trung bình', f"{kpis['avg_order_value']:,.0f} VNĐ",
         PALETTE['purple']),
    ]
    
    return [
        {'label': label, 'value': value, 'color': color}
        for label, value, color in card_specs
    ]


def display_kpi_cards(cards: list) -> None:
    """Display a row of KPI cards in the notebook.
    
    Parameters:
        cards (list): List of card definition dictionaries
    """
    cards_html = '<div style="display:flex; gap:15px; flex-wrap:wrap; margin:15px 0;">'
    for card in cards:
        color = card.get('color', PALETTE['primary'])
        cards_html += f"""
        <div style="flex:1; min-width:180px; background:white;
                    border-left:5px solid {color}; padding:15px;
                    border-radius:5px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
            <p style="margin:0; font-size:12px; color:#7f8c8d;
                      text-transform:uppercase; letter-spacing:0.04em;">
                {card['label']}</p>
            <p style="margin:8px 0 0 0; font-size:24px; font-weight:700;
                      color:#2B3A42;">{card['value']}</p>
        </div>
        """
    cards_html += '</div>'
    display(HTML(cards_html))


def plot_sales_time_series(
    sales: pd.DataFrame, save_fig: bool = True
) -> None:
    """Plot monthly revenue and gross profit trends from the sales dataset.
    
    Parameters:
        sales (pd.DataFrame): Sales dataframe with Date, Revenue, COGS columns
        save_fig (bool): Whether to save figure to disk
    """
    set_plot_style()
    df = sales.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date')
    df['Gross_Profit'] = df['Revenue'] - df['COGS']
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    
    
    monthly = df.groupby('YearMonth', as_index=False)[
        ['Revenue', 'COGS', 'Gross_Profit']
    ].sum()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(monthly['YearMonth'], monthly['Revenue'] / 1e9,
            label='Revenue', color=PALETTE['primary'], lw=2)
    ax.plot(monthly['YearMonth'], monthly['Gross_Profit'] / 1e9,
            label='Gross Profit', color=PALETTE['blue'], lw=2)
    ax.fill_between(monthly['YearMonth'], monthly['Revenue'] / 1e9,
                    alpha=0.12, color=PALETTE['primary'])
    ax.set_title('Xu hướng Doanh Thu và Gross Profit theo Tháng', fontsize=14)
    ax.set_ylabel('Giá trị (Tỷ VNĐ)')
    ax.set_xlabel('Thời gian')
    ax.set_xticks(monthly['YearMonth'].iloc[::max(1, len(monthly)//12)])
    ax.set_xticklabels(
        monthly['YearMonth'].iloc[::max(1, len(monthly)//12)],
        rotation=45, ha='right'
    )
    ax.legend()
    format_spines(ax)
    plt.tight_layout()
    if save_fig:
        fig.savefig(FIG_DIR / '01_sales_revenue_trend.png',
                   dpi=150, bbox_inches='tight')
    plt.show()


def plot_order_status_distribution(
    orders: pd.DataFrame, save_fig: bool = True
) -> None:
    """Plot order status distribution for transaction-level bottleneck analysis.
    
    Parameters:
        orders (pd.DataFrame): Orders dataframe with order_status column
        save_fig (bool): Whether to save figure to disk
    """
    set_plot_style()
    counts = orders['order_status'].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    
    color_map = {
        'cancelled': PALETTE['danger'],
        'returned': PALETTE['accent'],
    }
    colors = [color_map.get(status, PALETTE['secondary'])
              for status in counts.index]
    
    bars = ax.barh(counts.index, counts.values, color=colors, alpha=0.9)
    
    _label_hbars(ax, fmt_fn=lambda v: f"{int(v):,}")
    
    ax.set_title('Phân bố trạng thái đơn hàng')
    ax.set_xlabel('Số lượng đơn hàng')
    ax.set_ylabel('Trạng thái đơn')
    format_spines(ax)
    plt.tight_layout()
    if save_fig:
        fig.savefig(FIG_DIR / '02_order_status_distribution.png',
                   dpi=150, bbox_inches='tight')
    plt.show()


def plot_stockout_overstock_segment(
    inventory: pd.DataFrame, products: pd.DataFrame, save_fig: bool = True
) -> None:
    """Plot stockout and overstock rate by segment using inventory snapshots.
    
    Parameters:
        inventory (pd.DataFrame): Inventory dataframe
        products (pd.DataFrame): Products dataframe
        save_fig (bool): Whether to save figure to disk
    """
    set_plot_style()
    
    
    product_segment = products[[
        'product_id', 'segment'
    ]].rename(columns={'segment': 'product_segment'})
    
    merged = inventory.merge(product_segment, on='product_id', how='left')
    merged['segment'] = merged['segment'].fillna(
        merged['product_segment']
    )
    
    grouped = merged.groupby('segment').agg({
        'stockout_flag': 'mean',
        'overstock_flag': 'mean'
    }).reset_index()
    grouped = grouped.sort_values('stockout_flag', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(grouped))
    width = 0.35
    ax.bar(x - width/2, grouped['stockout_flag'] * 100, width,
           label='Stockout Rate', color=PALETTE['stockout'], alpha=0.9)
    ax.bar(x + width/2, grouped['overstock_flag'] * 100, width,
           label='Overstock Rate', color=PALETTE['overstock'], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped['segment'], rotation=45, ha='right')
    ax.set_title('Stockout và Overstock theo Segment')
    ax.set_ylabel('Tỷ lệ (%)')
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    format_spines(ax)
    plt.tight_layout()
    if save_fig:
        fig.savefig(FIG_DIR / '03_stockout_overstock_segment.png',
                   dpi=150, bbox_inches='tight')
    plt.show()


def plot_traffic_revenue_indexed(
    master_df: pd.DataFrame, traffic: pd.DataFrame, save_fig: bool = True
) -> None:
    """Plot indexed trends for sessions, revenue and orders.
    
    Parameters:
        master_df (pd.DataFrame): Master transaction dataframe
        traffic (pd.DataFrame): Web traffic dataframe
        save_fig (bool): Whether to save figure to disk
    """
    set_plot_style()
    daily_sales = master_df.groupby('order_day').agg({
        'order_id': 'nunique',
        'net_revenue': 'sum'
    }).reset_index()
    daily_sales.columns = ['order_day', 'orders', 'revenue']
    daily_sales['order_day'] = pd.to_datetime(daily_sales['order_day'])
    
    traffic_data = traffic[['date', 'sessions']].copy()
    traffic_data['date'] = pd.to_datetime(
        traffic_data['date'], errors='coerce'
    )
    
    merged = pd.merge(
        daily_sales, traffic_data,
        left_on='order_day', right_on='date', how='inner'
    ).sort_values('order_day')
    
    merged = merged.set_index(
        'order_day'
    )[['orders', 'revenue', 'sessions']].rolling(
        window=30, min_periods=15
    ).mean().dropna()
    
    indexed = merged.divide(merged.iloc[0]).multiply(100)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(indexed.index, indexed['sessions'], label='Sessions',
            color=PALETTE['neutral'], linestyle='--', alpha=0.8)
    ax.plot(indexed.index, indexed['revenue'], label='Revenue',
            color=PALETTE['primary'], lw=2)
    ax.plot(indexed.index, indexed['orders'], label='Orders',
            color=PALETTE['blue'], lw=2)
    ax.set_title('So sánh xu hướng Traffic, Revenue và Orders (Base = 100)')
    ax.set_ylabel('Chỉ số')
    ax.set_xlabel('Ngày')
    ax.legend()
    format_spines(ax)
    plt.tight_layout()
    if save_fig:
        fig.savefig(FIG_DIR / '04_traffic_revenue_indexed.png',
                   dpi=150, bbox_inches='tight')
    plt.show()


def estimate_stockout_impact(
    inventory: pd.DataFrame, products: pd.DataFrame, days_in_month: int = 30
) -> pd.DataFrame:
    """Estimate stockout opportunity cost using inventory and product data.
    
    Parameters:
        inventory (pd.DataFrame): Inventory dataframe
        products (pd.DataFrame): Products dataframe
        days_in_month (int): Days per month for daily average calculation
    Returns:
        pd.DataFrame: DataFrame with stockout loss estimates by product
    """
    merged = inventory.merge(
        products[[
            'product_id', 'product_name', 'category', 'segment', 'price', 'cogs'
        ]], on='product_id', how='left', suffixes=('_inv', '_prod')
    )
    
    
    for col in ['product_name', 'category', 'segment']:
        merged[col] = merged[f'{col}_inv'].fillna(merged[f'{col}_prod'])
    
    merged['avg_daily_revenue'] = (
        merged['units_sold'] * merged['price'] / days_in_month
    )
    merged['lost_revenue_est'] = (
        merged['stockout_days'] * merged['avg_daily_revenue']
    )
    merged['lost_margin_est'] = merged['lost_revenue_est'] * (
        ((merged['price'] - merged['cogs']) / merged['price']).fillna(0)
    )
    merged['year_month'] = merged['snapshot_date'].dt.to_period(
        'M'
    ).astype(str)
    
    return merged


def plot_stockout_loss_by_category(
    loss_df: pd.DataFrame, save_fig: bool = True
) -> None:
    """Plot estimated lost revenue by category due to stockout.
    
    Parameters:
        loss_df (pd.DataFrame): Stockout loss dataframe from estimate_stockout_impact
        save_fig (bool): Whether to save figure to disk
    """
    set_plot_style()
    grouped = loss_df.groupby('category').agg({
        'lost_revenue_est': 'sum'
    }).reset_index()
    grouped = grouped.sort_values('lost_revenue_est', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(grouped['category'], grouped['lost_revenue_est'] / 1e6,
            color=PALETTE['stockout'], alpha=0.9)
    _label_hbars(ax, fmt_fn=lambda v: _fmt_number(v / 1e6, 1))
    
    ax.set_title('Estimated Lost Revenue do Stockout theo Category')
    ax.set_xlabel('Lost revenue estimate (Triệu VNĐ)')
    ax.set_ylabel('Category')
    format_spines(ax)
    plt.tight_layout()
    if save_fig:
        fig.savefig(FIG_DIR / '05_stockout_loss_by_category.png',
                   dpi=150, bbox_inches='tight')
    plt.show()


def plot_top_products_lost_revenue(
    loss_df: pd.DataFrame, top_n: int = 20, save_fig: bool = True
) -> None:
    """Plot the top products by estimated lost revenue from stockout.
    
    Parameters:
        loss_df (pd.DataFrame): Stockout loss dataframe from estimate_stockout_impact
        top_n (int): Number of top products to display
        save_fig (bool): Whether to save figure to disk
    """
    set_plot_style()
    product_loss = loss_df.groupby(
        ['product_id', 'product_name']
    ).agg({'lost_revenue_est': 'sum'}).reset_index()
    top_products = product_loss.sort_values(
        'lost_revenue_est', ascending=True
    ).tail(top_n)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(top_products['product_name'],
            top_products['lost_revenue_est'] / 1e6,
            color=PALETTE['blue'], alpha=0.9)
    _label_hbars(ax, fmt_fn=lambda v: _fmt_number(v / 1e6, 1))
    
    ax.set_title(f'Top {top_n} Products theo Estimated Lost Revenue')
    ax.set_xlabel('Lost revenue estimate (Triệu VNĐ)')
    ax.set_ylabel('Product')
    format_spines(ax)
    plt.tight_layout()
    if save_fig:
        fig.savefig(FIG_DIR / '06_top_products_lost_revenue.png',
                   dpi=150, bbox_inches='tight')
    plt.show()


def compute_promotion_margins(master_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute margin comparisons for promo and non-promo transactions.
    
    Parameters:
        master_df (pd.DataFrame): Master transaction dataframe
    Returns:
        Dict[str, Any]: Dictionary with margin analysis results
    """
    def _margin_percent(df: pd.DataFrame) -> float:
        """Calculate margin percentage for a dataframe."""
        total_revenue = df['net_revenue'].sum()
        if total_revenue == 0:
            return 0.0
        return df['gross_profit'].sum() / total_revenue * 100

    clean = master_df[master_df['net_revenue'] > 0].copy()
    promo = clean[clean['has_promo'] == 1]
    no_promo = clean[clean['has_promo'] == 0]
    
    margin_all = _margin_percent(clean)
    margin_with_promo = _margin_percent(promo)
    margin_without_promo = _margin_percent(no_promo)

    by_type = promo.groupby(
        'promo_type'
    ).apply(_margin_percent).rename('margin').reset_index()
    by_type['count'] = promo.groupby(
        'promo_type'
    )['order_id'].count().values

    stackable = promo.groupby(
        ['promo_type', 'stackable_flag']
    ).apply(_margin_percent).reset_index(name='margin')
    stackable['count'] = promo.groupby(
        ['promo_type', 'stackable_flag']
    )['order_id'].count().values

    result = {
        'margin_all': margin_all,
        'margin_with_promo': margin_with_promo,
        'margin_without_promo': margin_without_promo,
        'by_type': by_type,
        'stackable_breakdown': stackable,
        'penetration_rate': (
            promo.shape[0] / clean.shape[0] * 100 if clean.shape[0] else 0.0
        ),
    }
    return result


def build_customer_rfm(master_df: pd.DataFrame) -> pd.DataFrame:
    """Build a profit-based RFM table and customer segments.
    
    Parameters:
        master_df (pd.DataFrame): Master transaction dataframe
    Returns:
        pd.DataFrame: Customer RFM segmentation dataframe
    """
    analysis_date = master_df['order_date'].max() + pd.Timedelta(days=1)
    
    customer_df = master_df.groupby('customer_id').agg({
        'order_date': lambda x: (analysis_date - x.max()).days,
        'order_id': 'nunique',
        'gross_profit': 'sum'
    }).rename(columns={
        'order_date': 'recency',
        'order_id': 'frequency',
        'gross_profit': 'monetary_profit'
    }).reset_index()
    
    customer_df['R'] = pd.qcut(
        customer_df['recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop'
    )
    customer_df['F'] = pd.qcut(
        customer_df['frequency'].rank(method='first'), 5,
        labels=[1, 2, 3, 4, 5], duplicates='drop'
    )
    customer_df['M'] = pd.qcut(
        customer_df['monetary_profit'], 5,
        labels=[1, 2, 3, 4, 5], duplicates='drop'
    )
    
    customer_df['rfm_score'] = (
        customer_df['R'].astype(str) + customer_df['F'].astype(str)
        + customer_df['M'].astype(str)
    )

    def _assign_segment(row: pd.Series) -> str:
        """Assign RFM segment based on scores."""
        if row['R'] >= 4 and row['F'] >= 4:
            return 'Champions'
        if row['R'] >= 3 and row['F'] >= 3:
            return 'Loyal Customers'
        if row['R'] >= 4 and row['F'] <= 2:
            return 'New Customers'
        if row['R'] <= 2 and row['F'] >= 4:
            return 'At Risk'
        if row['R'] <= 2 and row['F'] <= 2:
            return 'Lost'
        return 'Need Attention'

    customer_df['segment'] = customer_df.apply(_assign_segment, axis=1)
    return customer_df


def compute_customer_lifecycle(
    master_df: pd.DataFrame
) -> Dict[str, Any]:
    """Compute customer cohort metrics for lifecycle analysis.
    
    Parameters:
        master_df (pd.DataFrame): Master transaction dataframe
    Returns:
        Dict[str, Any]: Dictionary with lifecycle metrics
    """
    cohort = master_df[['customer_id', 'order_date', 'order_id']].copy()
    cohort['order_year'] = cohort['order_date'].dt.year
    
    
    first_purchase = cohort.groupby(
        'customer_id'
    )['order_year'].min().rename('first_year')
    cohort = cohort.merge(first_purchase, on='customer_id', how='left')
    cohort['is_new_customer'] = (cohort['order_year'] ==
                                 cohort['first_year']).astype(int)

    cohort_customers = cohort.groupby(
        ['customer_id', 'order_year']
    ).agg({'is_new_customer': 'max'}).reset_index()
    
    summary = cohort_customers.groupby('order_year').agg({
        'customer_id': 'nunique',
        'is_new_customer': 'sum'
    }).reset_index()
    summary.columns = ['order_year', 'total_customers', 'new_customers']

    
    order_counts = cohort.groupby('customer_id')[
        'order_id'
    ].nunique()
    one_time_rate = (order_counts == 1).mean() * 100 if len(
        order_counts
    ) else 0.0
    
    return {
        'one_time_customer_rate': one_time_rate,
        'cohort_by_year': summary.sort_values('order_year'),
    }


def plot_revenue_averages_by_period(
    dfs: Dict[str, pd.DataFrame], save_fig: bool = True
) -> None:
    """Plot average revenue by year, month, and week in 3-panel layout.
    
    Parameters:
        dfs (Dict[str, pd.DataFrame]): Dictionary of datasets
        save_fig (bool): Whether to save figure to disk
    """
    set_plot_style()
    sales_data = dfs['sales'].copy()
    sales_data['Date'] = pd.to_datetime(sales_data['Date'])
    
    
    sales_data['Year'] = sales_data['Date'].dt.year
    sales_data['Month'] = sales_data['Date'].dt.month
    sales_data['Week'] = sales_data['Date'].dt.isocalendar().week

    avg_by_year = sales_data.groupby('Year')['Revenue'].mean()
    avg_by_month = sales_data.groupby('Month')['Revenue'].mean()
    avg_by_week = sales_data.groupby('Week')['Revenue'].mean()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Doanh Thu Trung Bình Theo Năm / Tháng / Tuần',
                 fontsize=14, fontweight='bold', y=1.02)

    # Subplot 1: By Year
    axes[0].bar(avg_by_year.index, avg_by_year.values / 1e6,
                color=PALETTE['primary'], alpha=0.8, edgecolor='white')
    axes[0].set_title('Doanh Thu Trung Bình Theo Năm',
                      fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Năm')
    axes[0].set_ylabel('Doanh Thu (Triệu VNĐ)')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    
    _label_vbars(axes[0], fmt_fn=lambda v: f'{v:.2f}M')
    format_spines(axes[0])

    # Subplot 2: By Month
    axes[1].plot(avg_by_month.index, avg_by_month.values / 1e6,
                 marker='o', linewidth=2.5, markersize=8,
                 color=PALETTE['blue'],
                 label='Trung bình theo tháng')
    axes[1].fill_between(avg_by_month.index, avg_by_month.values / 1e6,
                         alpha=0.2, color=PALETTE['blue'])
    axes[1].set_title('Doanh Thu Trung Bình Theo Tháng',
                      fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Tháng')
    axes[1].set_ylabel('Doanh Thu (Triệu VNĐ)')
    axes[1].set_xticks(range(1, 13))
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best')
    format_spines(axes[1])

    # Subplot 3: By Week
    axes[2].plot(avg_by_week.index, avg_by_week.values / 1e6,
                 marker='s', linewidth=1.5, markersize=4,
                 color=PALETTE['accent'], alpha=0.7,
                 label='Trung bình theo tuần')
    axes[2].fill_between(avg_by_week.index, avg_by_week.values / 1e6,
                         alpha=0.15, color=PALETTE['accent'])
    axes[2].set_title('Doanh Thu Trung Bình Theo Tuần',
                      fontweight='bold', fontsize=12)
    axes[2].set_xlabel('Tuần trong năm')
    axes[2].set_ylabel('Doanh Thu (Triệu VNĐ)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='best')
    format_spines(axes[2])

    plt.tight_layout()
    if save_fig:
        fig.savefig(FIG_DIR / '00_average_revenue_trends.png',
                   dpi=150, bbox_inches='tight')
    plt.show()


def plot_promotion_margins_analysis(
    margin_results: Dict[str, Any], save_fig: bool = True
) -> None:
    """Plot 3-panel visualization for promotion margin analysis.
    
    Parameters:
        margin_results (Dict[str, Any]): Results from compute_promotion_margins
        save_fig (bool): Whether to save figure to disk
    """
    set_plot_style()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        'Phân Tích Khả Năng Sinh Lời (Gross Margin) Theo Khuyến Mãi',
        fontsize=16, fontweight='bold', y=1.05
    )

    # Panel 1: Margin with vs without Promo
    ax1 = axes[0]
    labels_1 = ['Không Promo', 'Có Promo']
    values_1 = [
        margin_results['margin_without_promo'],
        margin_results['margin_with_promo']
    ]
    colors_1 = [
        PALETTE['primary'] if v >= 0 else PALETTE['danger'] for v in values_1
    ]

    bars1 = ax1.bar(labels_1, values_1, color=colors_1, alpha=0.85,
                    width=0.5)
    ax1.set_title('1. Tác động của Promo tới Margin',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Gross Margin (%)')
    ax1.axhline(0, color='black', linewidth=1)
    _label_vbars(ax1, fmt_fn=lambda v: f'{v:.1f}%', offset=0.02)
    format_spines(ax1)

    # Panel 2: Margin by Promo Type
    ax2 = axes[1]
    df_type = margin_results['by_type']
    labels_2 = df_type['promo_type'].str.title().tolist()
    values_2 = df_type['margin'].tolist()
    colors_2 = [
        PALETTE['primary'] if v >= 0 else PALETTE['danger'] for v in values_2
    ]

    bars2 = ax2.bar(labels_2, values_2, color=colors_2, alpha=0.85,
                    width=0.5)
    ax2.set_title('2. Margin theo Loại Khuyến Mãi',
                  fontsize=13, fontweight='bold')
    ax2.axhline(0, color='black', linewidth=1)
    _label_vbars(ax2, fmt_fn=lambda v: f'{v:.1f}%', offset=0.02)
    format_spines(ax2)

    # Panel 3: Margin by Stackable Flag
    ax3 = axes[2]
    df_stack = margin_results['stackable_breakdown'].copy()

    def _create_stackable_label(row: pd.Series) -> str:
        """Create readable label for stackable flag."""
        p_type = str(row['promo_type']).title()
        s_flag = ('Stackable' if float(row['stackable_flag']) == 1.0
                  else 'No Stack')
        return f"{p_type}\n({s_flag})"

    df_stack['label'] = df_stack.apply(_create_stackable_label, axis=1)
    labels_3 = df_stack['label'].tolist()
    values_3 = df_stack['margin'].tolist()
    colors_3 = [
        PALETTE['primary'] if v >= 0 else PALETTE['danger'] for v in values_3
    ]

    bars3 = ax3.bar(labels_3, values_3, color=colors_3, alpha=0.85,
                    width=0.6)
    ax3.set_title('3. Ảnh hưởng của việc Cộng dồn (Stackable)',
                  fontsize=13, fontweight='bold')
    ax3.axhline(0, color='black', linewidth=1)
    _label_vbars(ax3, fmt_fn=lambda v: f'{v:.1f}%', offset=0.02)
    format_spines(ax3)

    plt.tight_layout()
    if save_fig:
        fig.savefig(FIG_DIR / '07_promotion_margin_analysis.png',
                   dpi=150, bbox_inches='tight')
    plt.show()


def plot_returns_cancellations_deepdive(
    returns: pd.DataFrame,
    products: pd.DataFrame,
    order_items: pd.DataFrame,
    reviews: pd.DataFrame,
    shipments: pd.DataFrame,
    orders: pd.DataFrame,
    save_fig: bool = True,
) -> None:
    """Dashboard 2x2: Returns, cancellations, and customer experience analysis.
    
    Parameters:
        returns (pd.DataFrame): Returns dataframe
        products (pd.DataFrame): Products dataframe
        order_items (pd.DataFrame): Order items dataframe
        reviews (pd.DataFrame): Reviews dataframe
        shipments (pd.DataFrame): Shipments dataframe
        orders (pd.DataFrame): Orders dataframe
        save_fig (bool): Whether to save figure to disk
    """
    set_plot_style()

    # Data Preparation
    ret_prod = returns.merge(
        products[['product_id', 'category', 'segment', 'size']],
        on='product_id'
    )

    ship_ord = shipments.copy()
    ship_ord['lead_time'] = (
        pd.to_datetime(ship_ord['delivery_date'])
        - pd.to_datetime(ship_ord['ship_date'])
    ).dt.days
    ship_ord = ship_ord[
        (ship_ord['lead_time'] > 0) & (ship_ord['lead_time'] < 60)
    ]
    rev_ship = reviews.merge(
        ship_ord[['order_id', 'lead_time']], on='order_id', how='inner'
    )

    # Figure Setup
    fig = plt.figure(figsize=(18, 12), layout='constrained')
    gs = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle('Trải nghiệm khách hàng và Rủi ro giao dịch',
                 fontsize=16, fontweight='bold', y=1.02)

    # (A) Pareto Chart: Return Reasons
    ax_a = fig.add_subplot(gs[0, 0])
    reason_ref = returns.groupby('return_reason')[
        'refund_amount'
    ].sum().sort_values(ascending=False)
    cum_pct = reason_ref.cumsum() / reason_ref.sum() * 100

    ax_a2 = ax_a.twinx()
    bars_a = ax_a.bar(
        range(len(reason_ref)), reason_ref.values / 1e6,
        color=[
            PALETTE['danger'] if i < 2 else PALETTE['accent']
            for i in range(len(reason_ref))
        ],
        alpha=0.85
    )
    ax_a2.plot(range(len(reason_ref)), cum_pct.values,
               color=PALETTE['blue'], lw=2.5, marker='o', markersize=6)

    ax_a2.axhline(80, color='gray', ls='--', lw=1.2, alpha=0.7)
    ax_a2.text(len(reason_ref) - 1, 82, '80%', fontsize=9,
               color='gray', fontweight='bold')

    ax_a.set_xticks(range(len(reason_ref)))
    ax_a.set_xticklabels(reason_ref.index, rotation=30, ha='right',
                         fontsize=10)
    ax_a.set_title('Lý do hoàn hàng', fontsize=13)
    ax_a.set_ylabel('Refund (Triệu VNĐ)',
                    color=PALETTE['danger'], fontweight='bold')
    ax_a2.set_ylabel('Cumulative %', color=PALETTE['blue'],
                     fontweight='bold')
    format_spines(ax_a, right_border=True)

    # (B) Return Rate Heatmap: Category × Size
    ax_b = fig.add_subplot(gs[0, 1])
    ret_qty = ret_prod.groupby(
        ['category', 'size']
    )['return_quantity'].sum()
    sold_qty = order_items.merge(
        products[['product_id', 'category', 'size']], on='product_id'
    ).groupby(['category', 'size'])['quantity'].sum()
    ret_rate = (ret_qty / sold_qty * 100).unstack().fillna(0)

    sns.heatmap(ret_rate, ax=ax_b, annot=True, fmt='.1f',
                cmap='Reds', linewidths=0.5,
                cbar_kws={'label': 'Return Rate (%)'})
    ax_b.set_title('Tỷ lệ hoàn hàng theo Catagory và Size', fontsize=13)
    ax_b.set_ylabel('Category')
    ax_b.set_xlabel('Size')

    # (C) Rating vs Lead Time Scatter
    ax_c = fig.add_subplot(gs[1, 0])
    if len(rev_ship) > 0:
        ax_c.scatter(rev_ship['lead_time'], rev_ship['rating'],
                     alpha=0.15, s=15, color=PALETTE['neutral'])

        z = np.polyfit(rev_ship['lead_time'], rev_ship['rating'], 1)
        x_reg = np.linspace(rev_ship['lead_time'].min(),
                            rev_ship['lead_time'].max(), 100)
        ax_c.plot(x_reg, np.poly1d(z)(x_reg),
                  color=PALETTE['danger'], lw=2.5)

        r, p = pearsonr(rev_ship['lead_time'], rev_ship['rating'])
        sig_text = "Significant" if p < 0.05 else "Not Significant"

        ax_c.text(0.05, 0.95,
                  f'Correlation (r) = {r:.3f}\np-value = {p:.4f}'
                  f'\n[{sig_text}]',
                  transform=ax_c.transAxes, va='top', fontsize=11,
                  fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='white',
                           edgecolor=PALETTE['danger'], alpha=0.8))

    ax_c.set_xlabel('Lead Time (ngày)')
    ax_c.set_ylabel('Rating (1–5)')
    ax_c.set_title('Tác động của thời gian giao hàng đến Rating',
                   fontsize=13)
    ax_c.axvline(5, color=PALETTE['danger'], ls='--', alpha=0.5,
                label='SLA = 5 Ngày')
    ax_c.legend()
    format_spines(ax_c)

    # (D) Cancel Rate by Payment Method
    ax_d = fig.add_subplot(gs[1, 1])

    cancel_rates = orders.groupby('payment_method').apply(
        lambda x: (x['order_status'] == 'cancelled').mean() * 100
    ).sort_values(ascending=False)

    colors_d = [
        PALETTE['danger'] if i == 0 else PALETTE['secondary']
        for i in range(len(cancel_rates))
    ]
    bars_d = ax_d.bar(cancel_rates.index, cancel_rates.values,
                      color=colors_d, alpha=0.85, width=0.6)

    ax_d.set_title('Tỷ lệ hủy đơn theo Phương thức thanh toán',
                   fontsize=13)
    ax_d.set_ylabel('Cancel Rate (%)')
    ax_d.set_xlabel('Payment Method')
    ax_d.set_xticks(range(len(cancel_rates)))
    ax_d.set_xticklabels(cancel_rates.index, rotation=0, fontsize=11)
    _label_vbars(ax_d, fmt_fn=lambda v: f'{v:.1f}%')
    format_spines(ax_d)

    if save_fig:
        fig.savefig(FIG_DIR / '08_returns_cancellations_deepdive.png',
                   dpi=150, bbox_inches='tight')
    plt.show()


def plot_revenue_leakage_impact(
    master_df: pd.DataFrame,
    returns: pd.DataFrame,
    shipments: pd.DataFrame,
    save_fig: bool = True,
) -> Dict[str, float]:
    """Calculate and visualize revenue leakage from cancellations and returns.
    
    Parameters:
        master_df (pd.DataFrame): Master transaction dataframe
        returns (pd.DataFrame): Returns dataframe
        shipments (pd.DataFrame): Shipments dataframe
        save_fig (bool): Whether to save figure to disk
    Returns:
        Dict[str, float]: Dictionary of leakage metrics
    """
    set_plot_style()

    # Calculate leakage metrics
    cancelled_df = master_df[master_df['order_status'] == 'cancelled']
    missed_revenue = cancelled_df['net_revenue'].sum()
    missed_profit = cancelled_df['gross_profit'].sum()

    total_refund = returns['refund_amount'].sum()
    returned_orders = returns['order_id'].unique()
    sunk_shipping = shipments[
        shipments['order_id'].isin(returned_orders)
    ]['shipping_fee'].sum()
    total_return_loss = total_refund + sunk_shipping

    # Display KPI cards
    leakage_kpis = [
        {
            'label': 'Doanh Thu Hụt (Hủy Đơn)',
            'value': f"{missed_revenue / 1e9:.2f} Tỷ",
            'color': PALETTE['danger'],
        },
        {
            'label': 'Gross Profit Lỡ Mất',
            'value': f"{missed_profit / 1e9:.2f} Tỷ",
            'color': PALETTE['danger'],
        },
        {
            'label': 'Thiệt Hại Hoàn Hàng',
            'value': f"{total_return_loss / 1e9:.2f} Tỷ",
            'color': PALETTE['accent'],
        },
        {
            'label': 'Phí Ship Lãng Phí',
            'value': f"{sunk_shipping / 1e6:.1f} Tr",
            'color': PALETTE['accent'],
        },
    ]
    display_kpi_cards(leakage_kpis)

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 3.5))
    labels = ['Doanh thu hụt (Hủy đơn)',
              'Thiệt hại dòng tiền (Hoàn hàng)']
    values = [missed_revenue, total_return_loss]
    colors = [PALETTE['danger'], PALETTE['accent']]

    bars = ax.barh(labels, [v / 1e9 for v in values], color=colors,
                   alpha=0.85, height=0.4)

    ax.set_title('Quy mô Thất thoát Tài chính (Revenue Leakage)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Giá trị (Tỷ VNĐ)')
    _label_hbars(ax, fmt_fn=lambda v: f"{v:.2f} Tỷ", offset=0.02)
    format_spines(ax)

    plt.tight_layout()
    if save_fig:
        fig.savefig(FIG_DIR / '09_revenue_leakage_impact.png',
                   dpi=150, bbox_inches='tight')
    plt.show()

    return {
        'missed_revenue': missed_revenue,
        'missed_profit': missed_profit,
        'total_refund': total_refund,
        'sunk_shipping': sunk_shipping,
        'total_return_loss': total_return_loss,
    }


def plot_inventory_health_overview(
    inv: pd.DataFrame, save_fig: bool = True
) -> None:
    """Dashboard: Inventory Health Overview with KPI cards and distributions.
    
    Parameters:
        inv (pd.DataFrame): Inventory dataframe
        save_fig (bool): Whether to save figure to disk
    """
    set_plot_style()

    # Configuration
    BACKGROUND = 'white'
    TEXT = '#2B3A42'
    MUTED = '#7f8c8d'
    c_ink = '#2B3A42'
    c_volatile = PALETTE['danger']

    def _kpi_card(ax: plt.Axes, title: str, value: str, subtitle: str,
                  color: str) -> None:
        """Draw KPI card on axes."""
        ax.axis('off')
        ax.plot([0.05, 0.05], [0.1, 0.9], color=color, lw=4,
               transform=ax.transAxes)
        ax.text(0.1, 0.70, title.upper(), fontsize=10, color=MUTED,
               fontweight='bold', transform=ax.transAxes)
        ax.text(0.1, 0.35, value, fontsize=24, color=TEXT,
               fontweight='bold', transform=ax.transAxes)
        ax.text(0.1, 0.10, subtitle, fontsize=10, color=MUTED,
               transform=ax.transAxes)

    def _clean_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str,
                   subtitle: Optional[str] = None) -> None:
        """Clean and format axes."""
        ax.set_title(title, fontsize=13, fontweight='bold',
                    loc='left', pad=15)
        if subtitle:
            ax.text(0, 1.02, subtitle, transform=ax.transAxes,
                   fontsize=10, color=MUTED)
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        format_spines(ax)

    # Create figure
    fig = plt.figure(figsize=(18, 10), facecolor=BACKGROUND)
    gs = gridspec.GridSpec(2, 15, figure=fig, height_ratios=[0.6, 2.4],
                           hspace=0.45, wspace=0.4)

    fig.suptitle('Inventory Health Overview', fontsize=18,
                fontweight='bold', color=TEXT, x=0.02, ha='left', y=0.99)
    fig.text(
        0.02, 0.935,
        'Baseline signals show whether inventory issue is shortage, '
        'excess, or misallocation.',
        fontsize=11, color=MUTED
    )

    # KPI Cards
    cards = [
        ('Stockout rate', _fmt_percent(inv['stockout_flag'].mean()),
         f"{_fmt_number(inv['stockout_days'].sum())} stockout days",
         PALETTE['stockout']),
        ('Overstock rate', _fmt_percent(inv['overstock_flag'].mean()),
         f"{_fmt_number(inv['overstock_flag'].sum())} overstock records",
         PALETTE['overstock']),
        ('Avg fill rate', _fmt_percent(inv['fill_rate'].mean()),
         'High avg, but local gaps remain', PALETTE['stable']),
        ('Avg days supply', _fmt_number(inv['days_of_supply'].mean(), 1),
         'Potential slow-moving inventory', PALETTE['warning']),
        ('Reorder flag', _fmt_percent(inv['reorder_flag'].mean()),
         'Not useful as current signal', PALETTE['neutral']),
    ]

    for idx, (title, value, subtitle, color) in enumerate(cards):
        ax_card = fig.add_subplot(gs[0, (idx * 3):(idx * 3) + 3])
        _kpi_card(ax_card, title, value, subtitle, color)

    # Distribution charts
    axes = [
        fig.add_subplot(gs[1, 0:5]),
        fig.add_subplot(gs[1, 5:10]),
        fig.add_subplot(gs[1, 10:15]),
    ]

    # Chart 1: Days of Supply
    p99_days = inv['days_of_supply'].quantile(0.99)
    days_clipped = inv['days_of_supply'].clip(upper=p99_days)
    sns.histplot(days_clipped, bins=50, kde=True, ax=axes[0],
                color=PALETTE['overstock'], alpha=0.75, edgecolor='white')
    axes[0].axvline(inv['days_of_supply'].median(), color=c_ink,
                   linestyle='--', linewidth=1.4,
                   label=f"Median {_fmt_number(inv['days_of_supply'].median())}")
    axes[0].axvline(inv['days_of_supply'].mean(), color=c_volatile,
                   linestyle='-', linewidth=1.4,
                   label=f"Mean {_fmt_number(inv['days_of_supply'].mean())}")
    _clean_axis(axes[0], 'Days of Supply Distribution',
               'Days of supply', 'Records',
               subtitle=f'Clipped at P99 = {_fmt_number(p99_days)} '
                        'to keep outliers readable')
    axes[0].legend(frameon=False, fontsize=9)

    # Chart 2: Fill Rate
    sns.histplot(inv['fill_rate'], bins=40, kde=True, ax=axes[1],
                color=PALETTE['stockout'], alpha=0.78, edgecolor='white')
    axes[1].axvline(inv['fill_rate'].mean(), color=c_volatile,
                   linewidth=1.5,
                   label=f"Mean {_fmt_percent(inv['fill_rate'].mean())}")
    _clean_axis(axes[1], 'Fill Rate Distribution', 'Fill rate', 'Records',
               subtitle='High average fill rate can hide SKU-level stockouts')
    axes[1].xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    axes[1].legend(frameon=False, fontsize=9)

    # Chart 3: Stockout Records
    stockout_counts = inv['stockout_flag'].value_counts().sort_index()
    bar_colors = [PALETTE['stable'], PALETTE['stockout']]
    axes[2].bar(['No stockout', 'Stockout'], stockout_counts.values,
               color=bar_colors, edgecolor='white', linewidth=1.1)
    _clean_axis(axes[2], 'Stockout Records', 'Stockout status', 'Records',
               subtitle='Count of product-month records by stockout flag')
    _label_vbars(axes[2])

    if save_fig:
        fig.savefig(FIG_DIR / '10_inventory_health_overview.png',
                   dpi=150, bbox_inches='tight')
    plt.show()


def plot_inventory_risk_concentration(
    inventory: pd.DataFrame, save_fig: bool = True
) -> None:
    """Dashboard 2x2: Inventory risk concentration by category and segment.
    
    Parameters:
        inventory (pd.DataFrame): Inventory dataframe
        save_fig (bool): Whether to save figure to disk
    """
    set_plot_style()

    BACKGROUND = 'white'
    TEXT = '#2B3A42'
    MUTED = '#7f8c8d'

    def _clean_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str,
                   subtitle: Optional[str] = None) -> None:
        """Clean and format axes."""
        ax.set_title(title, fontsize=13, fontweight='bold',
                    loc='left', pad=15)
        if subtitle:
            ax.text(0, 1.02, subtitle, transform=ax.transAxes,
                   fontsize=10, color=MUTED)
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        format_spines(ax)

    # Prepare data
    category_stockout = inventory.groupby('category').agg({
        'stockout_flag': 'mean',
        'overstock_flag': 'mean',
        'days_of_supply': 'mean',
    }).reset_index().sort_values('stockout_flag', ascending=False)

    segment_stockout = inventory.groupby('segment').agg({
        'stockout_flag': 'mean',
        'overstock_flag': 'mean',
        'days_of_supply': 'mean',
    }).reset_index().sort_values('stockout_flag', ascending=False)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor=BACKGROUND)
    fig.suptitle('Where Inventory Risk Concentrates',
                fontsize=18, fontweight='bold', color=TEXT,
                x=0.02, ha='left', y=1.02)
    fig.text(
        0.02, 0.975,
        'Category and segment cuts reveal if the root cause is '
        'broad replenishment policy or local SKU mix.',
        fontsize=10, color=MUTED
    )

    # Subplot 1: Stockout by Category
    sns.barplot(data=category_stockout, x='stockout_flag', y='category',
               ax=axes[0, 0], color=PALETTE['stockout'], edgecolor='white')
    _clean_axis(axes[0, 0], 'Stockout Rate by Category',
               'Stockout rate', 'Category')
    axes[0, 0].xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    _label_hbars(axes[0, 0], fmt_fn=_fmt_percent)

    # Subplot 2: Overstock by Category
    category_overstock = category_stockout.sort_values(
        'overstock_flag', ascending=False
    )
    sns.barplot(data=category_overstock, x='overstock_flag', y='category',
               ax=axes[0, 1], color=PALETTE['overstock'], edgecolor='white')
    _clean_axis(axes[0, 1], 'Overstock Rate by Category',
               'Overstock rate', 'Category')
    axes[0, 1].xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    _label_hbars(axes[0, 1], fmt_fn=_fmt_percent)

    # Subplot 3: Stockout by Segment
    segment_plot = segment_stockout.head(12)
    sns.barplot(data=segment_plot, x='stockout_flag', y='segment',
               ax=axes[1, 0], color=PALETTE['stockout'], edgecolor='white')
    _clean_axis(axes[1, 0], 'Stockout Rate by Segment',
               'Stockout rate', 'Segment')
    axes[1, 0].xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    _label_hbars(axes[1, 0], fmt_fn=_fmt_percent)

    # Subplot 4: Days of Supply by Segment
    segment_days = segment_stockout.sort_values(
        'days_of_supply', ascending=False
    ).head(12)
    sns.barplot(data=segment_days, x='days_of_supply', y='segment',
               ax=axes[1, 1], color=PALETTE['warning'], edgecolor='white')
    _clean_axis(axes[1, 1], 'Average Days of Supply by Segment',
               'Average days of supply', 'Segment')
    _label_hbars(axes[1, 1], fmt_fn=lambda v: _fmt_number(v))

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_fig:
        fig.savefig(FIG_DIR / '11_inventory_risk_concentration.png',
                   dpi=150, bbox_inches='tight')
    plt.show()


def plot_overstock_capital_locked(
    inventory: pd.DataFrame, products: pd.DataFrame, save_fig: bool = True
) -> None:
    """Dashboard: Inventory value locked in overstock by category.
    
    Parameters:
        inventory (pd.DataFrame): Inventory dataframe
        products (pd.DataFrame): Products dataframe
        save_fig (bool): Whether to save figure to disk
    """
    set_plot_style()

    BACKGROUND = 'white'
    TEXT = '#2B3A42'
    MUTED = '#7f8c8d'

    def _clean_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str,
                   subtitle: Optional[str] = None) -> None:
        """Clean and format axes."""
        ax.set_title(title, fontsize=14, fontweight='bold',
                    loc='left', pad=15)
        if subtitle:
            ax.text(0, 1.02, subtitle, transform=ax.transAxes,
                   fontsize=11, color=MUTED)
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        format_spines(ax)

    # Prepare data
    merged = inventory.merge(products[['product_id', 'cogs']],
                            on='product_id', how='left')
    overstock_df = merged[merged['overstock_flag'] == 1].copy()
    overstock_df['locked_capital'] = (
        overstock_df['stock_on_hand'] * overstock_df['cogs']
    )

    overstock_summary = overstock_df.groupby('category').agg({
        'locked_capital': 'sum'
    }).reset_index()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor=BACKGROUND)

    plot_df = overstock_summary.sort_values(
        'locked_capital', ascending=True
    ).copy()
    plot_df['share'] = (
        plot_df['locked_capital'] / plot_df['locked_capital'].sum()
    )

    colors = sns.light_palette(PALETTE['overstock'],
                              n_colors=len(plot_df) + 2)[2:]

    ax.barh(plot_df['category'], plot_df['locked_capital'],
           color=colors, edgecolor='white')

    _clean_axis(
        ax,
        'Inventory Value Locked in Overstock by Category',
        'Vốn bị khóa (tính theo COGS)',
        'Danh mục (Category)',
        subtitle='Locked capital highlights downside of blindly increasing stock.'
    )

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: _fmt_currency(x))
    )
    ax.margins(x=0.25)

    for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
        ax.text(
            patch.get_width(),
            patch.get_y() + patch.get_height() / 2,
            f" {_fmt_currency(row['locked_capital'])} "
            f"({_fmt_percent(row['share'])})",
            va='center', ha='left', fontsize=10, fontweight='bold',
            color=TEXT
        )

    plt.tight_layout()

    if save_fig:
        fig.savefig(FIG_DIR / '12_overstock_capital_locked.png',
                   dpi=150, bbox_inches='tight')
    plt.show()


def plot_stockout_recovery_scenarios(
    inventory: pd.DataFrame, products: pd.DataFrame, save_fig: bool = True
) -> pd.DataFrame:
    """Dashboard: Stockout recovery upside scenarios (naive linear estimate).
    
    Parameters:
        inventory (pd.DataFrame): Inventory dataframe
        products (pd.DataFrame): Products dataframe
        save_fig (bool): Whether to save figure to disk
    Returns:
        pd.DataFrame: Scenario analysis results
    """
    set_plot_style()

    BACKGROUND = 'white'
    TEXT = '#2B3A42'
    MUTED = '#7f8c8d'

    def _clean_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str,
                   subtitle: Optional[str] = None) -> None:
        """Clean and format axes."""
        ax.set_title(title, fontsize=14, fontweight='bold',
                    loc='left', pad=15)
        if subtitle:
            ax.text(0, 1.02, subtitle, transform=ax.transAxes,
                   fontsize=11, color=MUTED)
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        format_spines(ax)

    # Prepare data
    merged = inventory.merge(products[['product_id', 'price', 'cogs']],
                            on='product_id', how='left')

    merged['avg_daily_revenue'] = (
        merged['units_sold'] * merged['price'] / 30
    )
    merged['lost_revenue_est'] = (
        merged['stockout_days'] * merged['avg_daily_revenue']
    )
    merged['lost_margin_est'] = merged['lost_revenue_est'] * (
        ((merged['price'] - merged['cogs']) / merged['price']).fillna(0)
    )

    total_lost_revenue = merged['lost_revenue_est'].sum()
    total_lost_margin = merged['lost_margin_est'].sum()

    # Scenario analysis
    scenario_stockout = pd.DataFrame({
        'Mức giảm stockout giả định': [0.10, 0.20, 0.30, 0.40]
    })
    scenario_stockout['Recovered revenue estimate'] = (
        total_lost_revenue * scenario_stockout['Mức giảm stockout giả định']
    )
    scenario_stockout['Recovered gross margin estimate'] = (
        total_lost_margin * scenario_stockout['Mức giảm stockout giả định']
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5.8), facecolor=BACKGROUND)
    plot_df = scenario_stockout.copy()

    ax.bar(
        plot_df['Mức giảm stockout giả định'].map(lambda v: f'{v:.0%}'),
        plot_df['Recovered revenue estimate'],
        color=PALETTE['accent'], edgecolor='white', width=0.55
    )

    _clean_axis(
        ax,
        'Naive Upside Estimate from Stockout Reduction',
        'Assumed stockout reduction',
        'Recovered revenue estimate',
        subtitle='Simple linear estimate only; realistic scenario matrix next.'
    )

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: _fmt_currency(x))
    )
    _label_vbars(ax, fmt_fn=lambda v: _fmt_currency(v, 2))

    plt.tight_layout()

    if save_fig:
        fig.savefig(FIG_DIR / '13_stockout_recovery_naive_estimate.png',
                   dpi=150, bbox_inches='tight')
    plt.show()

    return scenario_stockout


def plot_customer_behavior_dashboard(
    customer_rfm: pd.DataFrame,
    customer_lifecycle: Dict[str, Any],
    save_fig: bool = True,
) -> None:
    """Dashboard 2x2: RFM segmentation and customer lifecycle analysis.
    
    Parameters:
        customer_rfm (pd.DataFrame): Customer RFM dataframe
        customer_lifecycle (Dict): Customer lifecycle metrics
        save_fig (bool): Whether to save figure to disk
    """
    set_plot_style()

    BACKGROUND = 'white'
    TEXT = '#2B3A42'
    MUTED = '#7f8c8d'

    # Prepare RFM aggregation
    rfm_agg = customer_rfm.groupby('segment').agg({
        'customer_id': 'count',
        'recency': 'mean',
        'monetary_profit': 'mean',
    }).reset_index()
    rfm_agg.columns = ['segment', 'count', 'avg_recency', 'avg_profit']
    rfm_agg = rfm_agg.sort_values('count', ascending=True)

    # Create figure
    fig = plt.figure(figsize=(18, 12), layout='constrained',
                    facecolor=BACKGROUND)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle('Phân Tích Hành Vi Khách Hàng: RFM & Vòng Đời',
                fontsize=18, fontweight='bold', color=TEXT)

    # (A) Customer Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.barh(rfm_agg['segment'], rfm_agg['count'],
                    color=PALETTE['blue'], alpha=0.85)
    ax1.set_title('(A) Phân bổ Khách hàng theo Phân khúc RFM',
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('Số lượng khách hàng')
    _label_hbars(ax1, fmt_fn=lambda v: f"{int(v):,}")
    format_spines(ax1)

    # (B) Quality Assessment: Recency vs Profit (Bubble)
    ax2 = fig.add_subplot(gs[0, 1])
    sizes = (rfm_agg['count'] / rfm_agg['count'].max()) * 2000 + 200

    scatter = ax2.scatter(rfm_agg['avg_recency'], rfm_agg['avg_profit'],
                         s=sizes, color=PALETTE['accent'], alpha=0.7,
                         edgecolor='white', linewidth=2)

    ax2.set_title('(B) Chất lượng Phân khúc: Avg Recency vs Avg Profit',
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Thời gian từ lần mua cuối - Avg Recency (Ngày)')
    ax2.set_ylabel('Lợi nhuận gộp trung bình - Avg Profit (VNĐ)')
    ax2.invert_xaxis()
    ax2.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: _fmt_currency(x))
    )

    for _, row in rfm_agg.iterrows():
        ax2.text(row['avg_recency'], row['avg_profit'],
                row['segment'], ha='center', va='center',
                fontsize=10, fontweight='bold', color=TEXT)
    format_spines(ax2)

    # (C) Customer Growth Stacked Bar
    ax3 = fig.add_subplot(gs[1, 0])
    cohort = customer_lifecycle['cohort_by_year']

    old_customers = cohort['total_customers'] - cohort['new_customers']
    x_years = cohort['order_year'].astype(str)

    ax3.bar(x_years, old_customers, label='Khách hàng Cũ (Returning)',
           color=PALETTE['neutral'], alpha=0.4)
    ax3.bar(x_years, cohort['new_customers'], bottom=old_customers,
           label='Khách hàng Mới (New)', color=PALETTE['primary'],
           alpha=0.85)

    ax3.set_title('(C) Tỷ trọng Khách Cũ vs Khách Mới qua các năm',
                 fontsize=14, fontweight='bold')
    ax3.set_ylabel('Số lượng Khách hàng')
    ax3.legend(loc='upper left')
    format_spines(ax3)

    # (D) One-time Purchase Rate Donut
    ax4 = fig.add_subplot(gs[1, 1])
    one_time = customer_lifecycle['one_time_customer_rate']
    returning = 100 - one_time

    wedges, texts, autotexts = ax4.pie(
        [one_time, returning],
        labels=['Mua 1 Lần (One-time)', 'Mua >= 2 Lần (Returning)'],
        autopct='%1.1f%%', startangle=90,
        colors=[PALETTE['danger'], PALETTE['stable']],
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=3)
    )
    plt.setp(autotexts, size=12, weight="bold", color="white")
    plt.setp(texts, size=11, weight="bold", color=TEXT)
    ax4.set_title(f'(D) Tỷ lệ Khách hàng Mua 1 lần',
                 fontsize=14, fontweight='bold')

    if save_fig:
        fig.savefig(FIG_DIR / '14_customer_behavior_dashboard.png',
                   dpi=150, bbox_inches='tight')
    plt.show()


def plot_annual_retention_trend(
    master_df: pd.DataFrame, save_fig: bool = True
) -> None:
    """Dashboard: Annual customer retention trend analysis.
    
    Parameters:
        master_df (pd.DataFrame): Master transaction dataframe
        save_fig (bool): Whether to save figure to disk
    """
    set_plot_style()

    BACKGROUND = 'white'
    TEXT = '#2B3A42'
    MUTED = '#7f8c8d'

    # Prepare data with vectorized operations
    cust_years = master_df[
        ['customer_id', 'order_year']
    ].drop_duplicates().copy()

    cust_years = cust_years.sort_values(['customer_id', 'order_year'])
    cust_years['next_active_year'] = cust_years.groupby(
        'customer_id'
    )['order_year'].shift(-1)
    cust_years['is_retained'] = (
        (cust_years['next_active_year'] ==
         cust_years['order_year'] + 1).astype(int)
    )

    annual_retention = cust_years.groupby(
        'order_year'
    )['is_retained'].mean() * 100

    # Remove final year (no data for next year)
    max_year = cust_years['order_year'].max()
    annual_retention = annual_retention[annual_retention.index < max_year]

    mean_retention = annual_retention.mean()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BACKGROUND)

    x = annual_retention.index.astype(str)
    y = annual_retention.values

    ax.fill_between(x, y, alpha=0.15, color=PALETTE['blue'])
    ax.plot(x, y, marker='o', markersize=8, color=PALETTE['blue'],
           linewidth=2.5, label='Retention Rate')

    ax.axhline(mean_retention, color=PALETTE['danger'],
              linestyle='--', linewidth=1.5, alpha=0.8)

    ax.text(len(x) - 0.5, mean_retention + 0.5,
           f"Trung bình: {mean_retention:.1f}%",
           color=PALETTE['danger'], fontweight='bold', ha='right',
           va='bottom', fontsize=11)

    for i, val in enumerate(y):
        ax.text(i, val + 1.2, f"{val:.1f}%", ha='center',
               va='bottom', fontsize=10, fontweight='bold',
               color=TEXT)

    ax.set_title('Xu hướng Giữ chân Khách hàng (Annual Retention Trend)',
                fontsize=15, fontweight='bold', loc='left', pad=15)
    ax.text(0, 1.02,
           'Tỷ lệ khách hàng mua sắm trong năm tiếp tục quay lại '
           'vào năm liền kề (Không tính năm cuối).',
           transform=ax.transAxes, fontsize=11, color=MUTED)

    ax.set_xlabel('Năm', fontweight='bold')
    ax.set_ylabel('Retention Rate (%)', fontweight='bold')

    format_spines(ax)
    ax.set_ylim(0, max(y) * 1.25)

    if save_fig:
        fig.savefig(FIG_DIR / '15_annual_retention_trend.png',
                   dpi=150, bbox_inches='tight')
    plt.show()


def plot_customer_survival_analysis(
    master_df: pd.DataFrame,
    customer_rfm: pd.DataFrame,
    save_fig: bool = True,
) -> None:
    """Dashboard: Customer survival curves by RFM segment.
    
    Parameters:
        master_df (pd.DataFrame): Master transaction dataframe
        customer_rfm (pd.DataFrame): Customer RFM dataframe
        save_fig (bool): Whether to save figure to disk
    """
    set_plot_style()

    BACKGROUND = 'white'
    TEXT = '#2B3A42'
    MUTED = '#7f8c8d'

    segment_colors = {
        'Champions': PALETTE['primary'],
        'Loyal Customers': PALETTE['blue'],
        'New Customers': PALETTE['secondary'],
        'At Risk': PALETTE['warning'],
        'Need Attention': PALETTE['accent'],
        'Lost': PALETTE['danger'],
    }

    # Calculate tenure
    customer_tenure = master_df.groupby('customer_id').agg({
        'order_date': ['min', 'max']
    }).reset_index()
    customer_tenure.columns = ['customer_id', 'first_buy', 'last_buy']

    customer_tenure['tenure_years'] = np.floor(
        (customer_tenure['last_buy'] - customer_tenure['first_buy']
         ).dt.days / 365
    ).astype(int)

    survival_data = customer_tenure.merge(
        customer_rfm[['customer_id', 'segment']], on='customer_id'
    )

    def _get_survival_rate(group: pd.DataFrame) -> pd.DataFrame:
        """Calculate survival rate for each year."""
        total = len(group)
        rates = []
        max_year = group['tenure_years'].max()
        for year in range(0, max_year + 1):
            survived = len(group[group['tenure_years'] >= year])
            rates.append({
                'year': year,
                'survival_rate': (survived / total) * 100
            })
        return pd.DataFrame(rates)

    survival_trend = survival_data.groupby(
        'segment'
    ).apply(_get_survival_rate).reset_index()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BACKGROUND)

    for segment in survival_trend['segment'].unique():
        seg_data = survival_trend[survival_trend['segment'] == segment]
        color = segment_colors.get(segment, PALETTE['neutral'])

        ax.step(seg_data['year'], seg_data['survival_rate'], where='post',
               lw=2.5, label=segment, color=color, alpha=0.85)

        last_point = seg_data.iloc[-1]
        ax.text(last_point['year'] + 0.1, last_point['survival_rate'],
               segment, color=color, fontweight='bold', fontsize=10,
               va='center')

    ax.set_title('SURVIVAL CURVES: Sau bao lâu thì khách hàng "bỏ cuộc"?',
                fontsize=15, fontweight='bold', loc='left', pad=15)
    ax.text(0, 1.02,
           'Đường cong biểu diễn xác suất khách hàng còn gắn bó '
           '(Survival Rate) sau N năm mua hàng đầu tiên.',
           transform=ax.transAxes, fontsize=11, color=MUTED)

    ax.set_xlabel('Số năm gắn bó (Tenure in years)', fontweight='bold')
    ax.set_ylabel('Xác suất còn ở lại (%)', fontweight='bold')

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 105)

    format_spines(ax)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    if save_fig:
        fig.savefig(FIG_DIR / '16_customer_survival_curves.png',
                   dpi=150, bbox_inches='tight')
    plt.show()


def plot_customer_triage_map(
    customer_rfm: pd.DataFrame, save_fig: bool = True
) -> None:
    """Dashboard: Customer triage map for strategic action assignment.
    
    Parameters:
        customer_rfm (pd.DataFrame): Customer RFM dataframe
        save_fig (bool): Whether to save figure to disk
    """
    set_plot_style()

    BACKGROUND = 'white'
    TEXT = '#2B3A42'
    MUTED = '#7f8c8d'

    segment_colors = {
        'Champions': PALETTE['primary'],
        'Loyal Customers': PALETTE['blue'],
        'New Customers': PALETTE['secondary'],
        'At Risk': PALETTE['warning'],
        'Need Attention': PALETTE['accent'],
        'Lost': PALETTE['danger'],
    }

    # Prepare data
    avg_recency_global = customer_rfm['recency'].mean()
    avg_frequency_global = customer_rfm['frequency'].mean()

    triage_data = customer_rfm.groupby('segment').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary_profit': 'sum',
        'customer_id': 'count',
    }).reset_index()
    triage_data.columns = [
        'segment', 'recency', 'frequency', 'monetary_profit', 'count'
    ]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=BACKGROUND)

    max_profit = triage_data['monetary_profit'].max()
    sizes = (triage_data['monetary_profit'] / max_profit) * 4000 + 300

    for idx, row in triage_data.iterrows():
        seg = row['segment']
        color = segment_colors.get(seg, PALETTE['neutral'])
        ax.scatter(row['recency'], row['frequency'], s=sizes[idx],
                  color=color, alpha=0.75, edgecolor='white', linewidth=2)

        ax.text(row['recency'], row['frequency'] + 0.12, seg,
               ha='center', va='bottom', fontsize=11,
               fontweight='bold', color=TEXT)

    # Add quadrant lines
    ax.axvline(avg_recency_global, color=MUTED, linestyle=':',
              linewidth=1.5)
    ax.axhline(avg_frequency_global, color=MUTED, linestyle=':',
              linewidth=1.5)

    y_limits = ax.get_ylim()
    x_limits = ax.get_xlim()

    ax.text(avg_recency_global + 2, y_limits[1] * 0.95,
           f"Avg Recency\n({avg_recency_global:.1f} ngày)",
           color=MUTED, ha='left', va='top', fontsize=10,
           fontweight='bold')
    ax.text(x_limits[0] * 0.95, avg_frequency_global + 0.05,
           f"Avg Frequency ({avg_frequency_global:.1f} đơn)",
           color=MUTED, ha='right', va='bottom', fontsize=10,
           fontweight='bold')

    ax.invert_xaxis()

    ax.set_title('CUSTOMER TRIAGE MAP: Phân loại chiến lược hành động '
                'cho từng nhóm',
                fontsize=15, fontweight='bold', loc='left', pad=15)
    ax.text(0, 1.03,
           'Kích thước bong bóng thể hiện Tổng Lợi Nhuận Gộp '
           '(Total Monetary Profit) của phân khúc.',
           transform=ax.transAxes, fontsize=11, color=MUTED)

    ax.set_xlabel('Recency (Số ngày kể từ lần mua cuối '
                 '- Càng nhỏ càng tốt)',
                 fontweight='bold')
    ax.set_ylabel('Frequency (Số đơn hàng trung bình)',
                 fontweight='bold')

    format_spines(ax)

    if save_fig:
        fig.savefig(FIG_DIR / '17_customer_triage_map.png',
                   dpi=150, bbox_inches='tight')
    plt.show()


def section_header(title: str, subtitle: str = '') -> None:
    """Display a professional section header in the notebook.
    
    Parameters:
        title (str): Main heading text
        subtitle (str): Optional subheading text
    """
    text = (f"<p style='margin:5px 0 0 0; color:#7f8c8d; "
            f"font-size:13px;'>{subtitle}</p>" if subtitle else '')
    html = f"""
    <div style='background: linear-gradient(135deg, #2D6A4F, #52B788);
                padding:18px; border-radius:8px; margin:20px 0;'>
        <h2 style='margin:0; color:white; font-size:22px;'>{title}</h2>
        {text}
    </div>
    """
    display(HTML(html))
