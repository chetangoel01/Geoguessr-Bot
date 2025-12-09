"""
Submission file generation and validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from ..config import GPS_BOUNDS


def create_submission(
    predictions: Dict[str, np.ndarray],
    template_path: str,
    output_path: str,
    validate: bool = True
) -> pd.DataFrame:
    """
    Create submission CSV from predictions.
    
    Args:
        predictions: Dictionary with predictions from predict_dataset
        template_path: Path to sample_submission.csv template
        output_path: Path to save submission
        validate: Whether to validate the submission
    
    Returns:
        Submission DataFrame
    """
    # Load template
    template = pd.read_csv(template_path)
    
    # Create submission dataframe
    submission = template.copy()
    
    # Sort predictions by sample_id to match template order
    sample_ids = predictions['sample_ids']
    sort_idx = np.argsort(sample_ids)
    
    # Fill in predictions
    top_k_states = predictions['top_k_states'][sort_idx]
    latitudes = predictions['latitudes'][sort_idx]
    longitudes = predictions['longitudes'][sort_idx]
    
    # State predictions
    submission['predicted_state_idx_1'] = top_k_states[:, 0]
    
    # Optional state predictions (fill with -1 if not enough)
    for k in range(1, 5):
        col = f'predicted_state_idx_{k+1}'
        if k < top_k_states.shape[1]:
            submission[col] = top_k_states[:, k]
        else:
            submission[col] = -1
    
    # GPS predictions
    submission['predicted_latitude'] = latitudes
    submission['predicted_longitude'] = longitudes
    
    # Validate
    if validate:
        is_valid, errors = validate_submission(submission)
        if not is_valid:
            print("Submission validation failed!")
            for error in errors:
                print(f"  - {error}")
            raise ValueError("Invalid submission")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    print(f"Submission saved to {output_path}")
    print(f"  Rows: {len(submission)}")
    print(f"  Columns: {list(submission.columns)}")
    
    return submission


def validate_submission(
    submission: pd.DataFrame,
    expected_rows: int = 16495
) -> tuple:
    """
    Validate submission against competition rules.
    
    Args:
        submission: Submission DataFrame
        expected_rows: Expected number of rows
    
    Returns:
        (is_valid, list of errors)
    """
    errors = []
    
    # Check row count
    if len(submission) != expected_rows:
        errors.append(f"Expected {expected_rows} rows, got {len(submission)}")
    
    # Check required columns
    required_columns = [
        'sample_id',
        'image_north', 'image_east', 'image_south', 'image_west',
        'predicted_state_idx_1',
        'predicted_state_idx_2', 'predicted_state_idx_3',
        'predicted_state_idx_4', 'predicted_state_idx_5',
        'predicted_latitude', 'predicted_longitude'
    ]
    
    missing_cols = set(required_columns) - set(submission.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # Check required fields are not null
    if submission['predicted_state_idx_1'].isna().any():
        errors.append("predicted_state_idx_1 has null values")
    
    if submission['predicted_latitude'].isna().any():
        errors.append("predicted_latitude has null values")
    
    if submission['predicted_longitude'].isna().any():
        errors.append("predicted_longitude has null values")
    
    # Check state index range (0-49)
    state_cols = [f'predicted_state_idx_{i}' for i in range(1, 6)]
    for col in state_cols:
        if col in submission.columns:
            values = submission[col].dropna()
            values = values[values != -1]  # -1 is valid for optional predictions
            
            if (values < 0).any() or (values > 49).any():
                invalid = values[(values < 0) | (values > 49)]
                errors.append(f"{col} has values outside [0, 49]: {invalid.unique()[:5]}")
    
    # Check latitude range (-90 to 90)
    lats = submission['predicted_latitude']
    if (lats < -90).any() or (lats > 90).any():
        errors.append(f"predicted_latitude outside [-90, 90]")
    
    # Check longitude range (-180 to 180)
    lons = submission['predicted_longitude']
    if (lons < -180).any() or (lons > 180).any():
        errors.append(f"predicted_longitude outside [-180, 180]")
    
    # Check for reasonable US coordinates
    # Continental US: lat 24-50, lon -125 to -66
    # With Alaska and Hawaii: lat 18-72, lon -180 to -65
    min_lat, max_lat = GPS_BOUNDS['lat']
    min_lon, max_lon = GPS_BOUNDS['lon']
    
    outside_us = ((lats < min_lat) | (lats > max_lat) | (lons > max_lon) | (lons < min_lon))
    if outside_us.any():
        n_outside = outside_us.sum()
        if n_outside > 0:
            # Warning, not error
            print(f"Warning: {n_outside} predictions may be outside US bounds")
    
    return len(errors) == 0, errors


def analyze_submission(
    submission: pd.DataFrame,
    ground_truth_path: Optional[str] = None
) -> Dict:
    """
    Analyze submission statistics.
    
    Args:
        submission: Submission DataFrame
        ground_truth_path: Optional path to ground truth for scoring
    
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    # State distribution
    state_counts = submission['predicted_state_idx_1'].value_counts()
    analysis['state_distribution'] = state_counts.to_dict()
    analysis['num_unique_states'] = len(state_counts)
    
    # GPS statistics
    analysis['lat_mean'] = submission['predicted_latitude'].mean()
    analysis['lat_std'] = submission['predicted_latitude'].std()
    analysis['lon_mean'] = submission['predicted_longitude'].mean()
    analysis['lon_std'] = submission['predicted_longitude'].std()
    
    # Top-k usage
    for k in range(2, 6):
        col = f'predicted_state_idx_{k}'
        if col in submission.columns:
            used = (submission[col] != -1).sum()
            analysis[f'positions_{k}_used'] = used
    
    # Confidence analysis (how often top prediction differs from 2nd)
    if 'predicted_state_idx_2' in submission.columns:
        same = (submission['predicted_state_idx_1'] == submission['predicted_state_idx_2']).sum()
        analysis['duplicate_top2'] = same
    
    # Geographic coverage
    analysis['lat_range'] = (
        submission['predicted_latitude'].min(),
        submission['predicted_latitude'].max()
    )
    analysis['lon_range'] = (
        submission['predicted_longitude'].min(),
        submission['predicted_longitude'].max()
    )
    
    return analysis


def compare_submissions(
    sub1_path: str,
    sub2_path: str
) -> Dict:
    """
    Compare two submissions.
    
    Args:
        sub1_path: Path to first submission
        sub2_path: Path to second submission
    
    Returns:
        Comparison statistics
    """
    sub1 = pd.read_csv(sub1_path)
    sub2 = pd.read_csv(sub2_path)
    
    comparison = {}
    
    # Top-1 agreement
    agree = (sub1['predicted_state_idx_1'] == sub2['predicted_state_idx_1']).mean()
    comparison['top1_agreement'] = agree
    
    # GPS difference
    lat_diff = np.abs(sub1['predicted_latitude'] - sub2['predicted_latitude']).mean()
    lon_diff = np.abs(sub1['predicted_longitude'] - sub2['predicted_longitude']).mean()
    comparison['mean_lat_diff'] = lat_diff
    comparison['mean_lon_diff'] = lon_diff
    
    # Samples where predictions differ
    differ_mask = sub1['predicted_state_idx_1'] != sub2['predicted_state_idx_1']
    comparison['n_different'] = differ_mask.sum()
    comparison['different_sample_ids'] = sub1.loc[differ_mask, 'sample_id'].tolist()[:20]
    
    return comparison
