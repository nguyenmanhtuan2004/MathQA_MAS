# 📊 Visualization Guide - MathQA Multi-Agent System

Detailed guide to the visualization and data analysis tools in the MathQA Multi-Agent System project.

## 🎯 Overview

This project provides a comprehensive visualization system to:
- Compare the performance of prompting methods
- Analyze detailed metrics (accuracy, latency, cost, tokens)
- Visualize results across different datasets
- Support decision-making and model optimization

## 🛠️ Visualization Structure

### 1. Main Tools
- **`__main__.py`**: Main command-line interface for visualization

### 2. Data Folders
- **`results/`**: Contains JSON files with experiment results
- **`save_log/`**: Stores detailed logs from runs

## 🎨 Supported Chart Types

### 1. Bar Charts with Dual-Axis
Display multiple metrics simultaneously:
- **Primary axis**: Accuracy (0-1)
- **Secondary axis**: Cost, Latency, or Tokens
- **Color coding**: Each method has a distinct color

### 2. Subplot Layout
- **Multi-dataset comparison**: Each dataset is a subplot
- **Shared legends**: Common legend for all subplots
- **Consistent scaling**: Y-axis is standardized

### 3. Interactive Features
- **Grid lines**: Help read exact values
- **Value annotations**: Show values on each bar
- **Professional styling**: Uses Seaborn with custom colors

## 🚀 Using the Command Line Interface

### 1. Compare Prompting Methods

#### Basic Syntax:
```bash
python __main__.py measure --methods [METHOD1] [METHOD2] ... --datasets [DATASET1] [DATASET2] ... --metric [METRIC]
```

#### Example Usage:

**Compare accuracy across all datasets:**
```bash
python __main__.py measure --methods Zero-shot CoT PoT PaL MultiAgent --datasets GSM8K TATQA TABMWP --metric tokens
```

**Analyze cost on a specific dataset:**
```bash
python __main__.py measure --methods PoT MultiAgent --datasets TATQA --metric total_cost
```

**Measure latency for main methods:**
```bash
python __main__.py measure --methods CoT PoT MultiAgent --datasets GSM8K TABMWP --metric latency_sec
```

### 2. Analyze Removal Experiments

```bash
python __main__.py remove --datasets GSM8K TATQA TABMWP --metric tokens
```

or to analyze all datasets:
```bash
python __main__.py remove --datasets all --metric total_cost
```

## 📈 Supported Metrics

### 1. `tokens`
- **Display**: Accuracy (bar) + Input/Output tokens (lines)
- **Application**: Analyze token usage patterns
- **Dual metrics**: Input tokens (red line), Output tokens (blue line)

### 2. `total_cost`
- **Display**: Accuracy (bar) + Total cost (red line)
- **Application**: Cost-benefit analysis
- **Unit**: USD

### 3. `latency_sec`
- **Display**: Accuracy (bar) + Response time (red line)
- **Application**: Performance optimization
- **Unit**: Seconds

## 🎨 Customization and Styling

### Color Palette
```python
highlight_colors = {
    'PaL': '#F28E2B',        # Vibrant orange
    'CoT': '#e74c3c',        # Bright red
    'Zero-shot': '#76B7B2',  # Light green
    'PoT': '#3498db',        # Blue
    'MultiAgent': "#74CBA2", # Mint green
}
```

### Layout Settings
- **Figure size**: 6×6 inches per subplot
- **Margins**: Optimized for readability
- **Legend position**: Centered at the bottom
- **Grid**: Y-axis dotted lines

## 📊 Exploratory Data Analysis (EDA)

### Jupyter Notebooks in the `EDA/` folder:

#### 1. `GSM8K_EDA.ipynb`
- Analysis of grade school math problems
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

### EDA Workflow:
1. **Data loading**: Load from the results folder
2. **Statistical summary**: Mean, std, distributions
3. **Error analysis**: Identify failure patterns
4. **Performance correlation**: Between metrics
5. **Visualization**: Custom plots and insights

## 🔧 Data Pipeline

### 1. Data Processing
- **JSON format**: Structured experiment results from prompting methods
- **CSV format**: Specialized data for removal experiments (MultiAgent-RMVe)
- **Metrics**: accuracy, latency, cost, tokens
- **Metadata**: run_id, error logs

### 2. Visualization Generation
- Data is loaded from the `results/` folder
- Charts are generated via `__main__.py` using the command line interface

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
- ✅ Load data efficiently with pandas
- ✅ Use appropriate figure sizes
- ✅ Optimize for both screen and print

## 🚨 Troubleshooting

### Common Issues:

#### 1. File Not Found Error
```
FileNotFoundError: File not found: results/PoT_GSM8K.json
```
**Solution**: Run the experiment first or check the file name

#### 2. Empty Plots
**Problem**: Subplot displays "No data"
**Solution**: Verify data format and column names

#### 3. Legend Overlap
**Problem**: Legend covers data
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
- [ ] Outlier detection and analysis
- [ ] Cost optimization recommendations

## 📞 Support

If you encounter issues with visualization:
1. Check file structure and data format
2. Verify command syntax
3. Review error logs in the terminal
4. Consult EDA notebooks for data insights

---

**📝 Note**: The visualization system is designed to be scalable and extensible. You can easily add new metrics, methods, or datasets to the existing framework.
