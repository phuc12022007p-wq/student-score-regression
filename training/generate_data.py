from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "student_scores.csv"

def main(n_samples: int = 500, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0, 20, n_samples)
    attendance = rng.uniform(50, 100, n_samples)
    assignments = rng.integers(0, 11, n_samples)
    previous = rng.uniform(0, 10, n_samples)
    
    noise = rng.normal(0, 0.55, n_samples)
    exam = np.clip(
        0.12 * hours + 0.025 * attendance + 0.18 * assignments + 0.42 * previous - 2 + noise,
        0,
        10
    )
    
    frame = pd.DataFrame({
        "hours_studied": hours.round(2),
        "attendance": attendance.round(2),
        "assignments_completed": assignments,
        "previous_score": previous.round(2),
        "exam_score": exam.round(2),
    })
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    print(f"Created {OUTPUT_PATH} with {len(frame)} rows")

if __name__ == "__main__":
    main()