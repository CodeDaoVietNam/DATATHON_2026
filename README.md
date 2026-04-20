# Datathon 2026 — Team [Tên Team]

## Cấu trúc thư mục
```bash 
datathon-2026/
├── data/raw/          # 15 file CSV gốc (không upload lên GitHub)
├── data/processed/    # Data sau khi clean
├── notebooks/         # Jupyter notebooks theo thứ tự
├── src/               # Python modules
├── figures/           # Charts cho report
├── submissions/       # submission.csv
└── report/            # report.pdf
```

## Cách chạy lại kết quả

# 1. Cài đặt
pip install -r requirements.txt

# 2. Đặt data vào đúng chỗ
cp path/to/data/*.csv data/raw/

# 3. Chạy theo thứ tự
jupyter nbconvert --to notebook --execute notebooks/01_data_quality.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_time_series_properties.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_model.ipynb

# 4. File submission xuất hiện tại
submissions/submission.csv

## Kết quả
- Kaggle MAE: [điền sau]
- Kaggle RMSE: [điền sau]  
- Kaggle R²: [điền sau]

## Team
- [Tên 1] — [Phần đảm nhận]
- [Tên 2] — [Phần đảm nhận]