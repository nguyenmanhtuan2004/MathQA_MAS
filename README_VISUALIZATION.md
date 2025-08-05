# 📊 Visualization Guide - MathQA Multi-Agent System

Hướng dẫn chi tiết về các công cụ visualization và phân tích dữ liệu trong project MathQA Multi-Agent System.

## 🎯 Tổng quan

Project này cung cấp hệ thống visualization toàn diện để:
- So sánh hiệu suất các phương pháp prompting
- Phân tích chi tiết metrics (accuracy, latency, cost, tokens)
- Trực quan hóa kết quả trên các dataset khác nhau
- Hỗ trợ ra quyết định và tối ưu hóa mô hình

## 🛠️ Cấu trúc Visualization

### 1. Công cụ chính
- **`__main__.py`**: Command-line interface chính cho visualization

### 2. Thư mục dữ liệu
- **`results/`**: Chứa file JSON kết quả từ các experiments
- **`save_log/`**: Lưu trữ logs chi tiết từ các runs

## 🎨 Các loại biểu đồ được hỗ trợ

### 1. Bar Charts với Dual-Axis
Hiển thị multiple metrics cùng lúc:
- **Primary axis**: Accuracy (0-1)
- **Secondary axis**: Cost, Latency, hoặc Tokens
- **Color coding**: Mỗi method có màu riêng biệt

### 2. Subplot Layout
- **Multi-dataset comparison**: Mỗi dataset một subplot
- **Shared legends**: Legend chung cho tất cả subplots
- **Consistent scaling**: Y-axis được chuẩn hóa

### 3. Interactive Features
- **Grid lines**: Hỗ trợ đọc giá trị chính xác
- **Value annotations**: Hiển thị giá trị trên mỗi bar
- **Professional styling**: Sử dụng Seaborn với custom colors

## 🚀 Sử dụng Command Line Interface

### 1. So sánh các phương pháp prompting

#### Cú pháp cơ bản:
```bash
python __main__.py measure --methods [METHOD1] [METHOD2] ... --datasets [DATASET1] [DATASET2] ... --metric [METRIC]
```

#### Ví dụ thực tế:

**So sánh accuracy trên tất cả datasets:**
```bash
python __main__.py measure --methods Zero-shot CoT PoT PaL MultiAgent --datasets GSM8K TATQA TABMWP --metric tokens
```

**Phân tích cost trên một dataset cụ thể:**
```bash
python __main__.py measure --methods PoT MultiAgent --datasets TATQA --metric total_cost
```

**Đo latency cho các phương pháp chính:**
```bash
python __main__.py measure --methods CoT PoT MultiAgent --datasets GSM8K TABMWP --metric latency_sec
```

### 2. Phân tích removal experiments

```bash
python __main__.py remove --datasets GSM8K TATQA TABMWP --metric tokens
```

hoặc để phân tích tất cả datasets:
```bash
python __main__.py remove --datasets all --metric total_cost
```

## 📈 Metrics được hỗ trợ

### 1. `tokens`
- **Hiển thị**: Accuracy (bar) + Input/Output tokens (lines)
- **Ứng dụng**: Phân tích token usage patterns
- **Dual metrics**: Input tokens (red line), Output tokens (blue line)

### 2. `total_cost`
- **Hiển thị**: Accuracy (bar) + Total cost (red line)
- **Ứng dụng**: Cost-benefit analysis
- **Đơn vị**: USD

### 3. `latency_sec`
- **Hiển thị**: Accuracy (bar) + Response time (red line)
- **Ứng dụng**: Performance optimization
- **Đơn vị**: Seconds

## 🎨 Customization và Styling

### Color Palette
```python
highlight_colors = {
    'PaL': '#F28E2B',        # Cam nổi bật
    'CoT': '#e74c3c',        # Đỏ rực
    'Zero-shot': '#76B7B2',  # Xanh lục nhẹ
    'PoT': '#3498db',        # Xanh dương
    'MultiAgent': "#74CBA2", # Xanh lá mint
}
```

### Layout Settings
- **Figure size**: 6×6 inches per subplot
- **Margins**: Optimized cho readability
- **Legend position**: Centered bottom
- **Grid**: Y-axis dotted lines

## 📊 Exploratory Data Analysis (EDA)

### Jupyter Notebooks trong thư mục `EDA/`:

#### 1. `GSM8K_EDA.ipynb`
- Phân tích grade school math problems
- Distribution analysis
- Error pattern identification

#### 2. `TATQA_EDA.ipynb`
- Table-based financial QA analysis
- Complex reasoning patterns
- Performance across question types

#### 3. `TABMWP_EDA.ipynb`
- Multi-modal math word problems
- Table and text integration analysis
- Visual-textual reasoning patterns

### Workflow EDA:
1. **Data loading**: Load từ results folder
2. **Statistical summary**: Mean, std, distributions
3. **Error analysis**: Identify failure patterns
4. **Performance correlation**: Giữa các metrics
5. **Visualization**: Custom plots và insights

## 🔧 Data Pipeline

### 1. Data Processing
- **JSON format**: Structured experiment results từ các phương pháp prompting
- **CSV format**: Specialized data cho removal experiments (MultiAgent-RMVe)
- **Metrics**: accuracy, latency, cost, tokens
- **Metadata**: run_id, error logs

### 2. Visualization Generation
- Dữ liệu được load từ thư mục `results/`
- Charts được tạo thông qua `__main__.py` với command line interface

## 📁 File Structure

```
MAS/
├── __main__.py                 # Main CLI for visualization
├── EDA/
│   ├── GSM8K_EDA.ipynb       # GSM8K analysis notebook
│   ├── TATQA_EDA.ipynb       # TATQA analysis notebook
│   └── TABMWP_EDA.ipynb      # TABMWP analysis notebook
├── results/
│   ├── PoT_TATQA.json        # Method results
│   ├── MultiAgent_TATQA.json
│   └── ...
└── save_log/
    ├── PoT_results_tatqa_*.json
    └── ...
```

## 🎯 Best Practices

### 1. Data Quality
- ✅ Ensure complete experiment data before plotting
- ✅ Check for missing files and handle gracefully
- ✅ Validate metric ranges for realistic values

### 2. Visualization Design
- ✅ Use consistent color schemes across plots
- ✅ Include error handling for missing data
- ✅ Provide clear axis labels and legends

### 3. Performance
- ✅ Load data efficiently với pandas
- ✅ Use appropriate figure sizes
- ✅ Optimize for both screen và print

## 🚨 Troubleshooting

### Common Issues:

#### 1. File Not Found Error
```
FileNotFoundError: Không tìm thấy file: results/PoT_GSM8K.json
```
**Solution**: Chạy experiment trước hoặc check file name

#### 2. Empty Plots
**Problem**: Subplot hiển thị "Không có dữ liệu"
**Solution**: Verify data format và column names

#### 3. Legend Overlap
**Problem**: Legend che khuất data
**Solution**: Adjust `bbox_to_anchor` parameters

### Debug Commands:
```bash
# Check available files
ls results/

# Validate JSON format
python -m json.tool results/PoT_TATQA.json

# Test basic plotting
python __main__.py measure --methods PoT --datasets TATQA --metric tokens
```

## 🔮 Future Enhancements

### Planned Features:
- [ ] Interactive Plotly dashboard
- [ ] Real-time experiment monitoring
- [ ] Automated report generation
- [ ] Statistical significance testing
- [ ] Export capabilities (PNG, PDF, SVG)

### Advanced Analytics:
- [ ] Correlation analysis between metrics
- [ ] Performance prediction models
- [ ] Outlier detection và analysis
- [ ] Cost optimization recommendations

## 📞 Support

Nếu gặp issues với visualization:
1. Check file structure và data format
2. Verify command syntax
3. Review error logs trong terminal
4. Consult EDA notebooks for data insights

---

**📝 Note**: Visualization system được thiết kế để scalable và extensible. Bạn có thể dễ dàng thêm new metrics, methods, hoặc datasets vào existing framework.
