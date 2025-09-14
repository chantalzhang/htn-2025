# Athlete Dataset Processing Pipeline

A comprehensive pipeline for processing and merging athlete statistics from multiple sports into a unified dataset.

## Directory Structure

```
athlete_dataset_pipeline/
├─ data/                    # Raw CSVs or JSONs for each sport
│   ├─ basketball_raw.csv
│   ├─ swimming_raw.csv
│   ├─ soccer_raw.csv
│   ├─ tennis_raw.csv
│   ├─ track_field_raw.csv
│   ├─ gymnastics_raw.csv
│   ├─ volleyball_raw.csv
│   ├─ baseball_raw.csv
│   ├─ hockey_raw.csv
│   └─ cycling_raw.csv
├─ processed/               # Processed per-sport CSVs
│   ├─ basketball.csv
│   └─ ...
├─ scripts/
│   ├─ process_sport.py     # Function to process one sport
│   ├─ merge_sports.py      # Script to merge all sports
│   └─ utils.py             # Helper functions (e.g., unit conversions)
├─ main.py                  # Entry point to run the full pipeline
├─ requirements.txt
└─ README.md
```

## Features

### Data Processing
- **Automatic unit conversion**: Heights to cm, weights to kg, times to seconds
- **Position standardization**: Consistent position names across sports
- **Data validation**: Quality checks and outlier detection
- **Missing data handling**: Smart filling and cleaning strategies
- **Derived metrics**: BMI calculation and other computed fields

### Supported Sports
1. Basketball
2. Swimming
3. Soccer
4. Tennis
5. Track & Field
6. Gymnastics
7. Volleyball
8. Baseball
9. Hockey
10. Cycling

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Full Pipeline
Process all sports and merge into unified dataset:
```bash
python main.py
```

### Process Single Sport
```bash
python main.py --sport basketball
```

### Merge Only
Skip processing and only merge existing processed files:
```bash
python main.py --merge-only
```

### Custom Output Directory
```bash
python main.py --output-dir /path/to/output
```

### Programmatic Usage

```python
from main import process_single_sport, quick_merge

# Process a single sport
result = process_single_sport("basketball")

# Quick merge all processed sports
merge_result = quick_merge()
```

## Input Data Format

Raw CSV files should contain athlete data with flexible column naming. The pipeline handles various column name conventions:

### Common Columns (flexible naming)
- **Name**: `name`, `player_name`, `athlete_name`, `full_name`
- **Height**: `height`, `ht`, `height_in`, `height_cm`
- **Weight**: `weight`, `wt`, `weight_lbs`, `weight_kg`
- **Age**: `age`, `years_old`
- **Position**: `position`, `pos`, `role`, `event`
- **Team**: `team`, `club`, `organization`
- **Country**: `country`, `nation`, `nationality`

### Sport-Specific Columns
Each sport can have additional performance metrics that will be automatically processed.

## Output

### Processed Individual Sports
- Standardized column names and units
- Cleaned and validated data
- Added derived metrics (BMI, etc.)
- Consistent formatting

### Merged Dataset
- `athlete_dataset_merged.csv`: Combined data from all sports
- `dataset_metadata.json`: Comprehensive dataset statistics and quality metrics
- `sport_comparison.json`: Cross-sport analysis and comparisons

## Data Quality Features

- **Completeness tracking**: Missing data percentages per column
- **Outlier detection**: Automatic flagging of unrealistic values
- **Unit validation**: Ensures consistent measurement units
- **Quality scoring**: Overall data quality percentage
- **Validation reports**: Detailed processing results

## Example Workflow

1. **Add raw data**: Place CSV files in `data/` directory with `_raw.csv` suffix
2. **Run pipeline**: Execute `python main.py`
3. **Review results**: Check processed files and quality reports
4. **Analyze dataset**: Use merged CSV and metadata for analysis

## Extensibility

The pipeline is designed to be easily extended:

- **New sports**: Add processing logic in `process_sport.py`
- **New metrics**: Extend `calculate_derived_metrics()` function
- **Custom validation**: Add rules in `validate_athlete_data()`
- **Output formats**: Modify merge functions for different output types

## Error Handling

- Graceful handling of missing files
- Detailed error reporting
- Partial processing support (continues with available data)
- Validation warnings for data quality issues
