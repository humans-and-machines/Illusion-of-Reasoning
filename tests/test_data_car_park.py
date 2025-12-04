from collections import Counter

from data.car_park import build_rush_small_balanced as rush


def test_rc_idx_and_idx_rc_inverse():
    size = 5
    row, col = 3, 1
    idx = rush.rc_idx(row, col, size)
    assert idx == row * size + col
    assert rush.idx_rc(idx, size) == (row, col)


def test_canonicalize_prioritizes_earliest_horizontal_two():
    size = 4
    grid = [None] * (size * size)
    state = rush.BoardPlacement(grid, size)
    cells_vertical = rush._place_v(state, 0, 0, 3, pid=9)
    cells_first_h2 = rush._place_h(state, 0, 1, 2, pid=2)  # earliest H2 -> becomes A
    cells_second_h2 = rush._place_h(state, 1, 2, 2, pid=5)
    pieces = {
        9: rush.Piece(9, cells_vertical, "V", 3),
        2: rush.Piece(2, cells_first_h2, "H", 2),
        5: rush.Piece(5, cells_second_h2, "H", 2),
    }

    canonical = rush._canonicalize(grid, pieces, size)

    assert canonical[1:3] == "AA"  # earliest horizontal length-2 is mapped to A
    assert canonical[0] == "B"  # first non-A piece keeps earliest non-A label
    assert canonical[6:8] == "CC"  # subsequent piece advances the label


def test_bfs_solve_finds_direct_solution():
    size = 4
    board = "AA" + "o" * (size * size - 2)  # A starts at columns 0–1 on the top row
    solvable, moves, path = rush.bfs_solve(board, size=size, max_nodes=100)
    assert solvable is True
    assert moves == 1
    assert path == ["A>2"]


def test_split_counts_preserves_bucket_size():
    n_tr, n_va, n_te = rush._split_counts(bucket_size=5, train_frac=0.8, val_frac=0.1, test_frac=0.1)
    assert n_tr + n_va + n_te == 5
    assert all(part >= 0 for part in (n_tr, n_va, n_te))


def test_stratified_split_keeps_bucket_totals_intact():
    rows = [
        {"size": 4, "difficulty": "easy", "id": "e1"},
        {"size": 4, "difficulty": "easy", "id": "e2"},
        {"size": 4, "difficulty": "hard", "id": "h1"},
        {"size": 4, "difficulty": "hard", "id": "h2"},
        {"size": 5, "difficulty": "medium", "id": "m1"},
        {"size": 5, "difficulty": "medium", "id": "m2"},
    ]
    train, val, test = rush.stratified_split(rows, split=(0.5, 0.25, 0.25), seed=7)

    def bucket_counts(items):
        counter = Counter()
        for row in items:
            counter[(row["size"], row["difficulty"])] += 1
        return counter

    original = bucket_counts(rows)
    combined = bucket_counts(train + val + test)

    assert combined == original  # no loss or duplication
    assert len(train) + len(val) + len(test) == len(rows)


def test_dedupe_and_filter_already_solved():
    size = 4
    goal_board = "ooAA" + "o" * (size * size - 4)  # A already at exit
    boards = ["repeat", "unique", goal_board]
    deduped, removed = rush.dedupe_boards(boards, seen={"repeat"})
    assert removed == 1
    assert "unique" in deduped and goal_board in deduped

    filtered, filtered_count = rush.filter_already_solved(deduped, size)
    assert filtered_count == 1
    assert goal_board not in filtered


def test_balance_by_size_and_difficulty_downsamples_evenly():
    rows = [
        {"size": 4, "difficulty": "easy", "id": "e1"},
        {"size": 4, "difficulty": "medium", "id": "m1"},
        {"size": 4, "difficulty": "medium", "id": "m2"},
        {"size": 4, "difficulty": "hard", "id": "h1"},
        {"size": 5, "difficulty": "easy", "id": "e5"},
    ]
    balanced = rush.balance_by_size_and_difficulty(rows, sizes=[4, 5], seed=0)
    size4 = [row for row in balanced if row["size"] == 4]
    size5 = [row for row in balanced if row["size"] == 5]

    assert len(size4) == 3  # min count across easy/medium/hard buckets
    assert len(size5) == 1  # untouched when buckets are incomplete
