import random
import copy
import json
import os
import dotenv

dotenv.load_dotenv()


def is_solvable(puzzle, N):
    # count inversions
    flat = [num for row in puzzle for num in row if num != 0]
    inversions = sum(
        1 for i in range(len(flat)) for j in range(i + 1, len(flat))
        if flat[i] > flat[j]
    )
    # blank row from bottom
    for i, row in enumerate(puzzle):
        if 0 in row:
            blank_row_from_bottom = N - i
            break
    # odd N: even inversions; even N: (inversions + blank_from_bottom) odd
    if N % 2 == 1:
        return inversions % 2 == 0
    return (inversions + blank_row_from_bottom) % 2 == 1


def get_valid_moves(position, N):
    moves = []
    row, col = position
    if row > 0: moves.append((-1, 0))
    if row < N - 1: moves.append((1, 0))
    if col > 0: moves.append((0, -1))
    if col < N - 1: moves.append((0, 1))
    return moves


def apply_move(position, move):
    return position[0] + move[0], position[1] + move[1]


def generate_puzzle(N, moves):
    solved_puzzle = [[N * i + j + 1 for j in range(N)] for i in range(N)]
    solved_puzzle[-1][-1] = 0
    blank_position = (N - 1, N - 1)

    puzzle = copy.deepcopy(solved_puzzle)
    move_sequence = []

    for _ in range(moves):
        valid_moves = get_valid_moves(blank_position, N)
        move = random.choice(valid_moves)
        new_blank_position = apply_move(blank_position, move)

        puzzle[blank_position[0]][blank_position[1]], puzzle[new_blank_position[0]][new_blank_position[1]] = \
            puzzle[new_blank_position[0]][new_blank_position[1]], puzzle[blank_position[0]][blank_position[1]]

        blank_position = new_blank_position
        move_sequence.append(move)

    return puzzle, move_sequence[::-1]


def print_puzzle(puzzle):
    for row in puzzle:
        print(' '.join(f'{num:2}' for num in row))
    print()


def load_instances_from_file(filename, instance_path="data/puzzle_instances"):
    base_path = os.getenv("BASE_PATH", "")
    full_path = os.path.join(base_path, instance_path, filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File '{full_path}' not found.")

    with open(full_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} instances from {full_path}")
    return data


def count_misplaced_tiles(puzzle):
    N = len(puzzle)
    misplaced = 0
    for r in range(N):
        for c in range(N):
            val = puzzle[r][c]
            if val == 0:
                continue
            goal_r = (val - 1) // N
            goal_c = (val - 1) % N
            if r != goal_r or c != goal_c:
                misplaced += 1
    return misplaced


if __name__ == "__main__":
    N = 10               # Puzzle size (e.g., 4 for 4x4)
    moves = 300           # Number of scrambling moves
    total_seeds = 100    # Number of seeds to generate

    base_path = os.getenv("BASE_PATH", "")
    output_dir = os.path.join(base_path, "puzzle_instances")
    os.makedirs(output_dir, exist_ok=True)

    size_label = f"{N}x{N}_{moves}"
    filename = os.path.join(output_dir, f"{size_label}_edu.json")

    all_instances = []

    for seed in range(total_seeds):
        random.seed(seed)
        puzzle, solution_moves = generate_puzzle(N, moves)

        if is_solvable(puzzle, N):
            instance = {
                "size": size_label,
                "seed": seed,
                "scramble_moves": moves,
                "puzzle": puzzle,
                "solution_moves": solution_moves,
                "misplaced": count_misplaced_tiles(puzzle)
            }

            all_instances.append(instance)

        else:
            print(f"Seed {seed} produced an unsolvable puzzle — skipped.")

    with open(filename, "w") as f:
        json.dump(all_instances, f, indent=2)

    print(f"\n✅ Saved {len(all_instances)} puzzles to {filename}")
