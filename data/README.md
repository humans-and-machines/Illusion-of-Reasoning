# Data directory
- `human_assessment_questions.txt`: prompt/assessment questions used for manual labeling; keep static and checked in here so scripts can read from `data/`.
- `car_park/`, `crossword/`: task-specific source data; caches or generated artifacts stay out of this tree (see `results/` for outputs).
