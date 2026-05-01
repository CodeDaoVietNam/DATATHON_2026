# 🏆 DATATHON 2026 — Vietnam Fashion E-Commerce Intelligence

> **Đội:** 404 Brain Not Found &nbsp;|&nbsp; **Cuộc thi:** Datathon 2026

---

## Mục lục

1. [Bối cảnh cuộc thi](#1-bối-cảnh-cuộc-thi)
2. [Mô tả bài toán](#2-mô-tả-bài-toán)
3. [Thành viên nhóm](#3-thành-viên-nhóm)
4. [Cách tải về và chạy dự án](#4-cách-tải-về-và-chạy-dự-án)
5. [Tech Stack](#5-tech-stack)
6. [Cấu trúc thư mục](#6-cấu-trúc-thư-mục)
7. [Insight rút ra được](#7-insight-rút-ra-được)
8. [Kết quả & So sánh Model](#8-kết-quả--so-sánh-model)
9. [Ràng buộc xử lý & Đặc điểm kỹ thuật](#9-ràng-buộc-xử-lý--đặc-điểm-kỹ-thuật)

---

## 1. Bối cảnh cuộc thi

**Datathon 2026** là cuộc thi phân tích dữ liệu dành cho sinh viên, tập trung vào lĩnh vực thương mại điện tử thời trang Việt Nam. Các đội thi được cung cấp bộ dữ liệu thực tế từ một sàn thương mại điện tử, bao gồm dữ liệu đơn hàng, sản phẩm, khuyến mãi, tồn kho, lưu lượng web, đánh giá khách hàng, và dữ liệu vận chuyển trải dài hơn **10 năm** (2012–2022).

**Mục tiêu tổng quát:** Tận dụng dữ liệu để xây dựng mô hình dự báo và cung cấp insight chiến lược cho doanh nghiệp, giúp tối ưu hóa vận hành và thúc đẩy tăng trưởng bền vững.

---

## 2. Mô tả bài toán

### Nhiệm vụ chính: Dự báo Doanh thu & Giá vốn (Forecasting)

Dựa trên dữ liệu lịch sử **Revenue** và **COGS** (Cost of Goods Sold) đến cuối năm 2022, nhóm cần:

- **Dự báo** Revenue và COGS theo từng ngày trong giai đoạn **2023-01-01 đến 2024-07-01** (548 ngày).
- Xuất kết quả ra file `submission.csv` đúng format của ban tổ chức.
- Đảm bảo mô hình **không bị data leakage** — chỉ sử dụng thông tin lịch sử, không dùng thông tin tương lai trong quá trình validation.

### Nhiệm vụ phụ: Phân tích & EDA Chiến lược

Song song với forecasting, nhóm thực hiện phân tích EDA sâu để rút ra insight kinh doanh:

- Phân tích xu hướng doanh thu, mùa vụ, và các sự kiện bất thường.
- Chẩn đoán hiệu quả chiến dịch khuyến mãi (Promotion Cannibalization).
- Phân tích chuỗi thời gian (Time Series Diagnostics): ACF/PACF, kiểm định dừng.
- Đánh giá rủi ro tồn kho & xây dựng Pre-Promotion Risk Score.
- Phân loại khách hàng theo mô hình RFM.

---

## 3. Thành viên nhóm

| Họ và Tên           | Vai trò                                                   |
| ------------------- | --------------------------------------------------------- |
| **Nguyễn Đức Tiến** | Forecasting Pipeline, Feature Engineering, Model Training |
| **Võ Ngọc Tiến**    | EDA & Time Series Diagnostics, Visualization              |
| **Nguyễn Kim Quốc** | Data Quality, Business Analysis, Reporting                |

> **Tên đội:** 404 Brain Not Found

---

## 4. Cách tải về và chạy dự án

### 4.1. Yêu cầu hệ thống

- Python **3.9+**
- Jupyter Notebook / JupyterLab
- RAM tối thiểu: **8 GB** (khuyến nghị 16 GB do kích thước dataset)

### 4.2. Clone repository

```bash
git clone https://github.com/CodeDaoVietNam/DATATHON_2026.git
cd DATATHON_2026
```

### 4.3. Cài đặt dependencies

```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv venv

# Kích hoạt môi trường ảo
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 4.4. Chuẩn bị dữ liệu

Đặt các file dữ liệu vào thư mục `dataset/`:

```
dataset/
├── sales.csv
├── sample_submission.csv
├── orders.csv
├── order_items.csv
├── products.csv
├── customers.csv
├── promotions.csv
├── inventory.csv
├── web_traffic.csv
├── reviews.csv
├── returns.csv
├── shipments.csv
└── geography.csv
```

### 4.5. Chạy Pipeline

**Khuyến nghị chạy theo thứ tự notebooks:**

```bash
# 1. Khởi động Jupyter
jupyter notebook

# 2. Chạy theo thứ tự:
#    notebooks/01_mcq.ipynb              → Multiple Choice Questions
#    notebooks/02_data_quality_exploration.ipynb → Data Quality
#    notebooks/03_Time_Series_Diagnostics.ipynb  → Time Series Analysis
#    notebooks/04_EDA_Final.ipynb                → EDA tổng hợp
#    notebooks/05_forecasting_model.ipynb        → Forecasting Pipeline chính
```

**Hoặc chạy trực tiếp notebook dự báo:**

```bash
jupyter notebook notebooks/05_forecasting_model.ipynb
# → Run All Cells
```

**Output:** File `dataset/submission.csv` chứa dự báo Revenue và COGS.

---

## 5. Tech Stack

### Ngôn ngữ & Môi trường

| Công cụ          | Phiên bản | Mục đích                         |
| ---------------- | --------- | -------------------------------- |
| Python           | 3.9+      | Ngôn ngữ lập trình chính         |
| Jupyter Notebook | —         | Môi trường phân tích & trình bày |

### Thư viện chính

| Thư viện                | Mục đích                                                               |
| ----------------------- | ---------------------------------------------------------------------- |
| **pandas**              | Xử lý & thao tác dữ liệu dạng bảng                                     |
| **numpy**               | Tính toán số học, mảng đa chiều                                        |
| **matplotlib**          | Vẽ biểu đồ cơ bản                                                      |
| **seaborn**             | Trực quan hóa thống kê nâng cao                                        |
| **scikit-learn**        | Machine Learning: ElasticNet, preprocessing, metrics, cross-validation |
| **statsmodels**         | Phân tích chuỗi thời gian: ACF/PACF, kiểm định ADF                     |
| **scipy**               | Tính toán thống kê                                                     |
| **lightgbm**            | Gradient Boosting (LightGBM) — Model chính                             |
| **xgboost**             | Gradient Boosting (XGBoost) — Model phụ                                |
| **prophet**             | Dự báo chuỗi thời gian theo mùa vụ (Meta)                              |
| **shap**                | Giải thích mô hình (Model Explainability)                              |
| **catboost**            | Gradient Boosting (CatBoost) — Model bổ sung                           |
| **jupyter / nbconvert** | Quản lý & xuất Notebook                                                |

---

## 6. Cấu trúc thư mục

```
DATATHON_2026/
│
├── notebooks/                       # Jupyter Notebooks theo pipeline
│   ├── 01_mcq.ipynb                    # Multiple Choice Questions
│   ├── 02_data_quality_exploration.ipynb  # Kiểm tra chất lượng dữ liệu
│   ├── 03_Time_Series_Diagnostics.ipynb   # Phân tích chuỗi thời gian
│   ├── 04_EDA_Final.ipynb              # EDA tổng hợp & insight kinh doanh
│   └── 05_forecasting_model.ipynb      # Pipeline dự báo chính (entry point)
│
├── src/                             # Source code được module hóa
│   ├── __init__.py
│   ├── eda_utils.py                    # Utilities cho EDA & visualization
│   └── utils/                          # Core utilities
│       ├── eda_utils.py                # EDA helpers (display, KPI cards, charts)
│       ├── utils.py                    # Plotting style, submission builder, metrics
│       ├── models.py                   # Model params (LightGBM, XGBoost, CatBoost)
│       ├── forecasting.py              # Forecasting pipeline chính
│       ├── feature_engineering.py      # Feature engineering cho time series
│       ├── diagnostics.py              # Validation diagnostics & explainability
│       └── cogs_ratio_experiments.py   # Thử nghiệm mô hình dự báo COGS ratio
│
├── dataset/                         # Dữ liệu đầu vào & output
│   ├── sales.csv                       # Target: Revenue & COGS lịch sử
│   ├── sample_submission.csv           # Template file nộp bài
│   ├── orders.csv                      # Dữ liệu đơn hàng
│   ├── order_items.csv                 # Chi tiết từng mặt hàng trong đơn
│   ├── products.csv                    # Danh mục sản phẩm
│   ├── customers.csv                   # Thông tin khách hàng
│   ├── promotions.csv                  # Chiến dịch khuyến mãi
│   ├── inventory.csv                   # Tồn kho theo ngày
│   ├── web_traffic.csv                 # Lưu lượng truy cập web
│   ├── reviews.csv                     # Đánh giá của khách hàng
│   ├── returns.csv                     # Dữ liệu hoàn trả
│   ├── shipments.csv                   # Vận chuyển & giao hàng
│   ├── geography.csv                   # Dữ liệu địa lý
│   └── baseline.ipynb                  # Notebook baseline mẫu
│
├── figures/                          # Hình ảnh & biểu đồ xuất ra
│   ├── 01_macro_kpi_dashboard.png      # Dashboard KPI tổng quan
│   ├── 02_hook_bottlenecks.png         # Phân tích điểm nghẽn
│   ├── 03_timeseries_decomposition.png # Phân rã chuỗi thời gian
│   ├── 04_stationarity_acf.png         # ACF/PACF & kiểm định dừng
│   ├── 05_dd1_inventory.png            # Deep Dive: Tồn kho
│   ├── 06_act2_curated_diagnostic.png  # Diagnostic tổng hợp
│   ├── 06_dd2_marketing_rfm.png        # Deep Dive: Marketing & RFM
│   └── 07_dd3_returns_reviews.png      # Deep Dive: Hoàn trả & Đánh giá
│
├── report/                           # Báo cáo chính thức
│   ├── main.tex                        # LaTeX source
│   ├── main.pdf                        # Báo cáo PDF xuất bản
│   └── references.bib                  # Tài liệu tham khảo
│
├── baseline.ipynb                    # Notebook baseline tham khảo (root)
├── requirements.txt                  # Danh sách thư viện
├── .gitignore                        # File loại trừ khỏi Git
└── README.md                         # File này
```

---

## 7. Insight rút ra được

### Khủng hoảng niềm tin — Cú gãy cấu trúc 2018

Tỷ lệ giữ chân khách hàng (Retention Rate) duy trì ổn định **>60%** trong 4 năm nhưng đột ngột sụt giảm hơn **14 điểm %** trong năm 2018 và **không bao giờ phục hồi**. Đây là dấu hiệu của một sự cố hệ thống nghiêm trọng (sự cố dịch vụ, chất lượng sản phẩm hoặc khủng hoảng truyền thông) chưa được giải quyết.

### Nghịch lý Chuỗi cung ứng — Thừa cái ế, thiếu cái cần

- **67%** trường hợp Stockout (hết hàng) xảy ra **song song** với **76%** trường hợp Overstock (dư hàng).
- Nhóm **Streetwear** khóa **76.9%** vốn lưu động (~42.3 tỷ VNĐ) trong hàng tồn dư, đồng thời gây thất thoát **80.9%** doanh thu tiềm năng do thiếu các SKU bán chạy.

### Bẫy Khuyến mãi — Promotion Trap

- Biên lợi nhuận gộp từ **+20%** lao dốc xuống **-14.5%** khi áp dụng khuyến mãi.
- **Fixed Discount** là "cỗ máy xay lợi nhuận" với mức lỗ kỷ lục **-63.3%**.
- Tồn tại hiện tượng **Promotion Cannibalization**: doanh thu tăng đột biến trong đợt khuyến mãi nhưng kéo theo sự sụt giảm mạnh ngay sau đó, cho thấy khuyến mãi chỉ dịch chuyển thời điểm mua hàng thay vì tạo ra nhu cầu mới.

### Rủi ro COD — Lỗ hổng dòng tiền

- Tỷ lệ hủy đơn COD chạm mức **16.0%**, gấp đôi các hình thức thanh toán trả trước.
- Doanh nghiệp đang gánh rủi ro tài chính lớn từ chi phí logistics ngược chiều.

### Mùa vụ & Xu hướng tăng trưởng

- Doanh thu tăng trưởng **+9.2% YoY** trong năm 2022.
- Mô hình mùa vụ rõ ràng: các sự kiện Tết, cuối quý, tháng 8 có spike doanh thu đặc trưng.
- Giai đoạn dự báo 548 ngày (2023-2024) đòi hỏi kỹ thuật **Recursive Forecasting** để tránh tích lũy sai số.

---

## 8. Kết quả & So sánh Model

### Metric đánh giá

| Metric | Ý nghĩa |
|---|---|
| **MAE** | Mean Absolute Error — Sai số tuyệt đối trung bình |
| **RMSE** | Root Mean Squared Error — Phạt nặng outlier |
| **R²** | Hệ số xác định — % phương sai được giải thích |

*Validation Set: Năm 2022 (365 ngày — holdout theo thời gian, không random split để tránh leakage)*

---

### Kết quả Baseline Models (Validation 2022)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Seasonal Naive** | 795,286 | 1,095,124 | 0.571 |
| **Monthly Median** | 1,321,645 | 1,643,697 | 0.033 |
| **Seasonal Growth** | 665,230 | 892,233 | 0.715 |

> Mọi model ML phức tạp phải vượt qua **Seasonal Growth Baseline (MAE: 665,230)** để chứng minh giá trị.

---

### Kết quả ML Models (Validation 2022 — Revenue)

| Model | MAE | RMSE | R² | Ghi chú |
|---|---|---|---|---|
| **LightGBM** | ~520,000 | ~710,000 | ~0.80 | Model chính, nhanh & hiệu quả |
| **XGBoost** | ~535,000 | ~730,000 | ~0.79 | Model phụ, ensemble |
| **CatBoost** | ~545,000 | ~740,000 | ~0.78 | Model bổ sung, xử lý tốt categorical |
| **ElasticNet** | ~680,000 | ~910,000 | ~0.70 | Linear baseline |
| **Ensemble (LGB+XGB+CAT)** | **~490,000** | **~670,000** | **~0.83** | Kết quả tốt nhất |

> **Scenario được chọn nộp bài:** `recovery100_spike_q1_aug_tet` — kết hợp Recovery Calibration 100% với spike uplift cho Tết, cuối quý 1 và tháng 8.

---

### Các kỹ thuật nâng cao

| Kỹ thuật | Mô tả |
|---|---|
| **Recursive Forecasting** | Dự báo từng ngày, dùng prediction ngày trước làm lag cho ngày sau |
| **Seasonal Growth Baseline** | Baseline mùa vụ có điều chỉnh tăng trưởng, dùng làm anchor |
| **Monthly Calibration** | Điều chỉnh prediction theo mean lịch sử từng tháng |
| **Event Spike Calibration** | Uplift/Calibration cho Tết (35%), Pre-Sale Events (22%), Aug End (18%) |
| **Recovery Trend Calibration** | Phục hồi trend sau COVID-19 (strength = 1.00) |
| **Component Shape Forecasting** | Dự báo dựa trên shape pattern lịch sử, không phụ thuộc vào lag |
| **COGS Ratio Model** | Dự báo COGS thông qua tỷ lệ COGS/Revenue theo tháng, không dự báo độc lập |
| **SHAP Explainability** | Giải thích tầm quan trọng của từng feature trong model |

---

## 9. Ràng buộc xử lý & Đặc điểm kỹ thuật

### Data Cleaning

| Vấn đề | Giải pháp |
|---|---|
| **COGS > Revenue (382 rows)** | Phát hiện và swap COGS/Revenue — dữ liệu bị nhập ngược |
| **Promo ID bị null** | Treat as `"No_Promo"` — missingness là tín hiệu kinh doanh có chủ đích |
| **Stacked Promo bị null** | Treat as `"No_Stacked_Promo"` — tránh high-cardinality near-empty feature |
| **Applicable Category null** | Treat as `"Sitewide"` — 80% promotion là Sitewide |

### Chống Data Leakage

- **Không dùng random split:** Validation phải là block theo thời gian (train ≤ 2021, valid = 2022).
- **Lag features:** Chỉ dùng lag đủ dài để không vi phạm time boundary (lag ≥ 7, rolling ≥ 14).
- **Calibration chỉ dùng lịch sử:** Monthly calibration, event spike calibration đều dựa trên dữ liệu train, không dùng validation/test.

### Feature Engineering

| Nhóm Feature | Chi tiết |
|---|---|
| **Lag Features** | Lag 7, 14, 21, 28, 56, 91, 182, 364 ngày của Revenue & COGS |
| **Rolling Statistics** | Mean, Std của cửa sổ 7, 14, 28, 56, 91 ngày |
| **Calendar Features** | Day of week, Day of month, Month, Quarter, Is Weekend, Is Holiday VN |
| **Event Features** | Is Tet, Pre/Post Tet, Is Sale Event Day, Pre/Post Sale Event, Q1 End Spike, Aug End Spike |
| **Promotion Profile** | Số lượng/cường độ khuyến mãi active theo ngày |
| **Traffic Profile** | Lưu lượng web theo ngày & nguồn truy cập |
| **Inventory Profile** | Fill rate, Stockout signal theo ngày |
| **Business Seasonality** | Seasonal component shape từ STL decomposition |

### Hyperparameters chính

```python
# LightGBM
LGBM_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}

# Recursive Forecast Settings
RECURSIVE_BASELINE_BLEND = 0.10   # 10% blend với seasonal baseline
RECURSIVE_EWM_ALPHA      = 0.85   # Smoothing factor cho EWM

# Event Spike Calibration Strengths
EVENT_SPIKE = {
    "tet": 0.35, "pre_tet": 0.12,
    "pre_sale_event": 0.22,
    "q1_end_spike": 0.10, "aug_end_spike": 0.18,
}
```

### Giới hạn & Lưu ý

- **Horizon 548 ngày** là rất dài cho recursive forecast — sai số có thể tích lũy theo thời gian.
- **CatBoost** cần cài đặt thêm riêng (không có trong `requirements.txt` mặc định), nếu không có thì pipeline tự động fallback về 2-model ensemble (LGB + XGB).
- Dataset rất lớn (~100MB+), cần đủ RAM khi load toàn bộ.

---

## Tài liệu tham khảo

- Hyndman, R.J. & Athanasopoulos, G. (2021). _Forecasting: Principles and Practice_ (3rd ed.)
- Chen, T. & Guestrin, C. (2016). _XGBoost: A Scalable Tree Boosting System_
- Ke, G. et al. (2017). _LightGBM: A Highly Efficient Gradient Boosting Decision Tree_
- Lundberg, S.M. & Lee, S.I. (2017). _A Unified Approach to Interpreting Model Predictions (SHAP)_

---

<div align="center">

**Made with ❤️ by Team 404 Brain Not Found**

_Datathon 2026 — Vietnam Fashion E-Commerce Intelligence_

</div>
