# OpenAi_Codex

This repository now includes a small utility script to normalise the `time` column in CSV files. The script removes timezone information (e.g. `+00:00`) so the format matches entries like `2025-09-09 06:29:00` from `XAUUSD_data.csv`.

## Usage

```
python convert_time_format.py XAUUSD_data_10min.csv
```

The command overwrites the input file with the `time` column formatted as `YYYY-MM-DD HH:MM:SS`. Use the `-o` option to write to a different file.
