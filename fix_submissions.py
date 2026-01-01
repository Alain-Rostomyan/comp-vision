"""
Quick script to convert submission CSV labels from numeric to string format.
"""
import pandas as pd
from pathlib import Path

# Label mapping
IDX_TO_LABEL = {
    0: 'apple',
    1: 'facebook',
    2: 'google',
    3: 'messenger',
    4: 'mozilla',
    5: 'samsung',
    6: 'whatsapp'
}

SUBMISSION_DIR = Path(__file__).parent / "outputs" / "submissions"

def fix_submission(filepath: Path):
    """Convert numeric labels to string labels in a submission CSV."""
    df = pd.read_csv(filepath)
    
    # Check if labels are already strings
    if df['Label'].dtype == 'object' and df['Label'].iloc[0] in IDX_TO_LABEL.values():
        print(f"  {filepath.name}: Already has string labels, skipping.")
        return
    
    # Map numeric labels to strings
    df['Label'] = df['Label'].map(IDX_TO_LABEL)
    
    # Save back
    df.to_csv(filepath, index=False)
    print(f"  {filepath.name}: Fixed!")

def main():
    print("Converting submission labels from numeric to string...\n")
    
    submission_files = list(SUBMISSION_DIR.glob("submission_*.csv"))
    
    if not submission_files:
        print("No submission files found!")
        return
    
    print(f"Found {len(submission_files)} submission file(s):\n")
    
    for filepath in submission_files:
        fix_submission(filepath)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
