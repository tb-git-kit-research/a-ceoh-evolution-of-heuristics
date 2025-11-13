import heapq
import json
import os
import time
import itertools
from copy import deepcopy
from typing import List, Tuple, Optional
from puzzle_generator import load_instances_from_file  # Your generator in educational format

Position = Tuple[int, int]
Puzzle = List[List[int]]


class PuzzleNode:
    def __init__(self, puzzle: Puzzle, g: int, parent=None, move=None):
        self.puzzle = puzzle
        self.N = len(puzzle)
        self.g = g
        self.h = self.manhattan_with_linear_conflict()
        self.f = self.g + self.h
        self.parent = parent
        self.move = move  # Move taken from parent (U, D, L, R)

    def __lt__(self, other):
        return self.f < other.f

    def manhattan_with_linear_conflict(self) -> int:
        dist = 0
        linear_conflict = 0
        N = self.N

        # Manhattan + row conflicts
        for r in range(N):
            row_conflicts = []
            for c in range(N):
                val = self.puzzle[r][c]
                if val == 0:
                    continue
                goal_r = (val - 1) // N
                goal_c = (val - 1) % N
                dist += abs(r - goal_r) + abs(c - goal_c)

                if r == goal_r:
                    row_conflicts.append(goal_c)
            for i in range(len(row_conflicts)):
                for j in range(i + 1, len(row_conflicts)):
                    if row_conflicts[i] > row_conflicts[j]:
                        linear_conflict += 1

        # Column conflicts
        for c in range(N):
            col_conflicts = []
            for r in range(N):
                val = self.puzzle[r][c]
                if val == 0:
                    continue
                goal_r = (val - 1) // N
                goal_c = (val - 1) % N

                if c == goal_c:
                    col_conflicts.append(goal_r)
            for i in range(len(col_conflicts)):
                for j in range(i + 1, len(col_conflicts)):
                    if col_conflicts[i] > col_conflicts[j]:
                        linear_conflict += 1

        return dist + 2 * linear_conflict

    def find_blank(self) -> Position:
        for r in range(self.N):
            for c in range(self.N):
                if self.puzzle[r][c] == 0:
                    return r, c

    def get_neighbors(self) -> List['PuzzleNode']:
        neighbors = []
        blank_r, blank_c = self.find_blank()
        directions = {
            (-1, 0): 'U',
            (1, 0): 'D',
            (0, -1): 'L',
            (0, 1): 'R'
        }

        for (dr, dc), label in directions.items():
            new_r, new_c = blank_r + dr, blank_c + dc
            if 0 <= new_r < self.N and 0 <= new_c < self.N:
                new_puzzle = deepcopy(self.puzzle)
                new_puzzle[blank_r][blank_c], new_puzzle[new_r][new_c] = \
                    new_puzzle[new_r][new_c], new_puzzle[blank_r][blank_c]
                neighbors.append(PuzzleNode(new_puzzle, self.g + 1, parent=self, move=label))
        return neighbors

    def is_goal(self):
        flat = [num for row in self.puzzle for num in row]
        return flat == list(range(1, self.N * self.N)) + [0]  # ✅ Educational format goal

    def serialize(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(tuple(row) for row in self.puzzle)


def solve_astar(start_puzzle: Puzzle, timeout: float = 60.0) -> Tuple[Optional[int], List[str], int, float]:
    open_set = []
    visited = set()
    counter = itertools.count()
    nodes_expanded = 0

    root = PuzzleNode(start_puzzle, g=0)
    heapq.heappush(open_set, (root.f, next(counter), root))
    visited.add(root.serialize())

    start_time = time.perf_counter()

    while open_set:
        if time.perf_counter() - start_time > timeout:
            return None, [], nodes_expanded, timeout

        _, _, current = heapq.heappop(open_set)
        nodes_expanded += 1

        if current.is_goal():
            move_seq = []
            node = current
            while node.parent is not None:
                move_seq.append(node.move)
                node = node.parent
            elapsed_time = time.perf_counter() - start_time
            return current.g, move_seq[::-1], nodes_expanded, elapsed_time

        for neighbor in current.get_neighbors():
            state = neighbor.serialize()
            if state in visited:
                continue
            visited.add(state)
            heapq.heappush(open_set, (neighbor.f, next(counter), neighbor))

    elapsed_time = time.perf_counter() - start_time
    return None, [], nodes_expanded, elapsed_time


def print_puzzle(puzzle: Puzzle):
    for row in puzzle:
        print(" ".join(f"{num:2}" for num in row))
    print()


if __name__ == "__main__":
    input_filename = "10x10_200_edu.json"
    base_path = os.getenv("BASE_PATH", "")
    instance_dir = os.path.join(base_path, "data", "puzzle_instances")

    input_path = os.path.join(instance_dir, input_filename)
    output_filename = input_filename.replace(".json", "_mh_lc.json")
    output_path = os.path.join(instance_dir, output_filename)

    puzzles = load_instances_from_file(input_filename)[:20]  # Run on first 20

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            solved_data = json.load(f)
            solved_seeds = {entry["seed"] for entry in solved_data}
    else:
        solved_data = []
        solved_seeds = set()

    total_skipped = 0
    total_solved = 0

    for sample in puzzles:
        seed = sample["seed"]
        if seed in solved_seeds:
            total_skipped += 1
            continue

        puzzle = sample["puzzle"]
        moves, move_seq, expanded, runtime = solve_astar(puzzle, timeout=600.0)

        result = {
            "seed": seed,
            "size": sample["size"],
            "scramble_moves": sample["scramble_moves"],
            "puzzle_astar_edu": puzzle,
            "solution_length": moves,
            "move_sequence": move_seq,
            "nodes_expanded": expanded,
            "runtime_seconds": round(runtime, 6),
            "timeout": (moves is None)
        }

        solved_data.append(result)
        total_solved += 1

        status = "⏱️ Timed out" if result["timeout"] else f"✅ Solved in {moves} moves"
        print(f"{status} | Seed {seed} | Expanded: {expanded} | Time: {runtime:.3f}s")

    with open(output_path, "w") as f:
        json.dump(solved_data, f, indent=2)

    print(f"\n📁 Results saved to: {output_path}")
    print(f"🔁 Skipped {total_skipped} already-solved instances")
    print(f"🆕 Solved {total_solved} new instances")
