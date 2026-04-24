#!/usr/bin/env python3
"""
NCU Report Comparison Script

This script finds the latest two ncu-rep files in /dli/task/ncu/reports
and compares key metrics between them.
"""

import os
import re
from datetime import datetime
from pathlib import Path
import ncu_report


def parse_filename_timestamp(filename):
    """
    Parse timestamp from filename with format: step1_yyyy-mm-ddTHH:MM:SS.ncu-rep

    Args:
        filename: The filename to parse

    Returns:
        datetime object or None if parsing fails
    """
    # Match pattern: step1_yyyy-mm-ddTHH:MM:SS.ncu-rep
    pattern = r'step1_(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.ncu-rep'
    match = re.match(pattern, filename)

    if match:
        timestamp_str = match.group(1)
        try:
            return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')
        except ValueError:
            return None
    return None


def find_latest_reports(reports_dir, count=2):
    """
    Find the latest N ncu-rep files in the specified directory.

    Args:
        reports_dir: Path to the reports directory
        count: Number of latest reports to return

    Returns:
        List of tuples (datetime, filepath) sorted by timestamp (newest first)
    """
    reports_path = Path(reports_dir)

    if not reports_path.exists():
        raise FileNotFoundError(f"Reports directory not found: {reports_dir}")

    # Find all .ncu-rep files and parse their timestamps
    report_files = []
    for file in reports_path.glob('step1_*.ncu-rep'):
        timestamp = parse_filename_timestamp(file.name)
        if timestamp:
            report_files.append((timestamp, file))

    if len(report_files) < count:
        raise ValueError(f"Found only {len(report_files)} report(s), need at least {count}")

    # Sort by timestamp (newest first)
    report_files.sort(key=lambda x: x[0], reverse=True)

    return report_files[:count]


def extract_metrics(kernel):
    """
    Extract key metrics from a kernel action.

    Args:
        kernel: IAction object representing the kernel

    Returns:
        Dictionary containing the extracted metrics
    """
    metrics = {}

    duration_metric = kernel.metric_by_name('gpu__time_duration.sum')
    if duration_metric:
        metrics['duration'] = {
            'value': duration_metric.value(),
            'unit': duration_metric.unit(),
            'name': 'Duration'
        }

    sm_throughput_metric = kernel.metric_by_name('sm__throughput.avg.pct_of_peak_sustained_elapsed')
    if sm_throughput_metric:
        metrics['sm_throughput'] = {
            'value': sm_throughput_metric.value(),
            'unit': sm_throughput_metric.unit(),
            'name': 'Compute (SM) Throughput'
        }

    memory_throughput_metric = kernel.metric_by_name('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed')
    if memory_throughput_metric:
        metrics['memory_throughput'] = {
            'value': memory_throughput_metric.value(),
            'unit': memory_throughput_metric.unit(),
            'name': 'Memory Throughput'
        }

    fma_active_metric = kernel.metric_by_name('sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed')
    if fma_active_metric:
        metrics['fma_active'] = {
            'value': fma_active_metric.value(),
            'unit': fma_active_metric.unit(),
            'name': 'FMA Cycles Active'
        }

    fp64_active_metric = kernel.metric_by_name('sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed')
    if fp64_active_metric:
        metrics['fp64_active'] = {
            'value': fp64_active_metric.value(),
            'unit': fp64_active_metric.unit(),
            'name': 'FP64 Cycles Active'
        }

    return metrics


def compare_reports(report_paths):
    """
    Load and compare metrics from multiple reports.

    Args:
        report_paths: List of tuples (datetime, filepath)

    Returns:
        Dictionary containing comparison data
    """
    comparison = {}

    for idx, (timestamp, filepath) in enumerate(report_paths):
        # Load the report
        report = ncu_report.load_report(str(filepath))

        # Get the first kernel from the first range
        if report.num_ranges() == 0:
            print(f"Warning: No ranges found in {filepath.name}")
            continue

        range_obj = report.range_by_idx(0)
        if range_obj.num_actions() == 0:
            print(f"Warning: No actions found in {filepath.name}")
            continue

        kernel = range_obj.action_by_idx(0)
        kernel_name = kernel.name()

        metrics = extract_metrics(kernel)

        comparison[f"report_{idx + 1}"] = {
            'timestamp': timestamp,
            'filepath': filepath,
            'kernel_name': kernel_name,
            'metrics': metrics
        }

    return comparison


def print_comparison(comparison):
    """
    Print a formatted comparison of the metrics in a table format.

    Args:
        comparison: Dictionary containing comparison data
    """

    # Get the two reports
    report_keys = sorted(comparison.keys())
    if len(report_keys) < 2:
        print("Not enough reports to compare")
        return

    report1 = comparison[report_keys[0]]
    report2 = comparison[report_keys[1]]

    # Prepare table data
    table_rows = []

    # Add basic metrics
    for metric_key in ['duration', 'sm_throughput', 'memory_throughput', 'fma_active', 'fp64_active']:
        if metric_key in report1['metrics'] and metric_key in report2['metrics']:
            m1 = report1['metrics'][metric_key]
            m2 = report2['metrics'][metric_key]

            v1 = m1['value']
            v2 = m2['value']

            # Calculate difference
            diff_str = "N/A"
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v2 != 0:
                diff_pct = ((v1 - v2) / v2) * 100
                diff_str = f"{diff_pct:+.2f}%"

            table_rows.append({
                'metric': m1['name'],  # Use the label for named metrics
                'report1': f"{v1:,.2f} {m1['unit']}",
                'report2': f"{v2:,.2f} {m2['unit']}",
                'change': diff_str
            })

    # Print table
    if table_rows:
        print("\n" + "=" * 120)

        # Calculate column widths
        metric_width = max(len(row['metric']) for row in table_rows)
        metric_width = max(metric_width, len("Metric"))

        report1_width = max(len(row['report1']) for row in table_rows)
        report1_width = max(report1_width, len("Report 1 (Newer)"))

        report2_width = max(len(row['report2']) for row in table_rows)
        report2_width = max(report2_width, len("Report 2 (Older)"))

        change_width = max(len(row['change']) for row in table_rows)
        change_width = max(change_width, len("Change"))

        # Print header
        header = f"{'Metric':<{metric_width}} | {'Report 1 (Newer)':>{report1_width}} | {'Report 2 (Older)':>{report2_width}} | {'Change':>{change_width}}"
        separator = "-" * len(header)

        print(header)
        print(separator)

        # Print rows
        for row in table_rows:
            print(f"{row['metric']:<{metric_width}} | {row['report1']:>{report1_width}} | {row['report2']:>{report2_width}} | {row['change']:>{change_width}}")

        print("=" * 120)


def main():
    """Main function to execute the comparison."""
    reports_dir = 'reports'

    try:
        # Find the latest two reports
        report_files = find_latest_reports(reports_dir, count=2)

        # Compare the reports
        comparison = compare_reports(report_files)

        # Print the comparison
        print_comparison(comparison)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())

