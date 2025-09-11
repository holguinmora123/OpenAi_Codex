import pandas as pd
from pathlib import Path

def convert_time_format(input_file: str, output_file: str | None = None) -> Path:
    """Convert the `time` column of a CSV to 'YYYY-MM-DD HH:MM:SS'.

    Parameters
    ----------
    input_file: str
        Path to the CSV file whose `time` column may contain timezone info.
    output_file: str | None, optional
        Path to save the converted CSV. If omitted, the input file is overwritten.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    df = pd.read_csv(input_file)
    if 'time' not in df.columns:
        raise ValueError("CSV must contain a 'time' column")

    # Parse dates and drop timezone info if present, then format like 'YYYY-MM-DD HH:MM:SS'
    times = pd.to_datetime(df['time'])
    times = times.dt.tz_localize(None)  # Remove timezone if it exists
    df['time'] = times.dt.strftime('%Y-%m-%d %H:%M:%S')

    output_path = Path(output_file) if output_file else Path(input_file)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize the time column format in a CSV")
    parser.add_argument("input_file", help="CSV file with a time column")
    parser.add_argument("-o", "--output-file", help="Where to save the converted CSV", default=None)
    args = parser.parse_args()

    out_path = convert_time_format(args.input_file, args.output_file)
    print(f"Converted CSV written to {out_path}")
