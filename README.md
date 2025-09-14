# 🧠 Mental Health Analytics Platform

## Hệ thống Phân tích Sức khỏe Tinh thần Người lao động

## 🏗️ Kiến trúc hệ thống

```
project_nhu/
├── 📁 app.py                     # Flask application chính
├── 📁 templates/
│   └── index.html               # Single Page Application (HTML, CSS, JS)
│── data.csv                     # Dataset
├── 📄 requirements.txt          # Python dependencies
└── 📄 README.md                 # Documentation
```

## 🚀 Cài đặt & Triển khai

### 📋 Yêu cầu hệ thống
- **Python**: 3.8 trở lên
- **RAM**: Tối thiểu 4GB (8GB khuyến nghị)
- **Storage**: 2GB dung lượng trống
- **Browser**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+

### 🔧 Hướng dẫn cài đặt chi tiết

#### Bước 1: Clone và setup môi trường
```bash
# Clone repository
git clone https://github.com/luuvannhathao/cho_ynhu.git
cd project_nhu

# Tạo virtual environment
python -m venv mental_health_env

# Kích hoạt virtual environment
# Windows:
mental_health_env\Scripts\activate
# macOS/Linux:
source mental_health_env/bin/activate
```

#### Bước 2: Cài đặt dependencies
```bash
# Cài đặt Python packages
pip install --upgrade pip
pip install -r requirements.txt

```

#### Bước 3: Chạy ứng dụng
```bash
python app.py
python3 app.py (nếu chạy kh được python app.py)
```


### 🌐 Truy cập ứng dụng
- **Local Development**: http://localhost:5000


## 📊 Dataset Requirements

### 📋 Required Columns
| Column | Type | Description |
|--------|------|-------------|
| `Survey_Date` | String | Ngày khảo sát (YYYY-MM-DD) |
| `Age` | Integer | Tuổi (18-65) |
| `Gender` | String | Giới tính |
| `Region` | String | Khu vực địa lý |
| `Industry` | String | Ngành nghề |
| `Job_Role` | String | Vai trò công việc |
| `Work_Arrangement` | String | Remote/Onsite/Hybrid |
| `Hours_Per_Week` | Integer | Giờ làm việc/tuần |
| `Mental_Health_Status` | String | Excellent/Good/Fair/Poor |
| `Burnout_Level` | String | Low/Medium/High |
| `Work_Life_Balance_Score` | Integer | Điểm 1-10 |
| `Physical_Health_Issues` | String | Yes/No |
| `Social_Isolation_Score` | Integer | Điểm 1-10 |
| `Salary_Range` | String | Khoảng lương |


## 📞 Support & Contact

### 🆘 Hỗ trợ kỹ thuật
- **GitHub Issues**: [luuvannhathao](https://github.com/your-repo/issues)
- **Số điện thoại**: gọi ngoài giờ hành chính

