import inspect


class GetPrompts():
    def __init__(self):

        # Name of the function implementing this heuristic
        self.prompt_func_name = "score_state"

        import solver.ceoh.problems.descriptions.puzzle.puzzle_astar_edu.run as run
        self.prompt_func_context = (f"The heuristic {self.prompt_func_name} will be used in the following A* procedure: \n"
                                    f"{inspect.getsource(run.astar_puzzle_core)} \n"
                                    f"{inspect.getsource(run.PuzzleNode.is_goal)} \n"
                                    f"{inspect.getsource(run.PuzzleNode.get_neighbors)} \n"
                                    f"{inspect.getsource(run.PuzzleNode.reconstruct_path)} \n"
                                    f"{inspect.getsource(run.PuzzleNode.get_objective_value)} \n"
                                    "Analyze how the heuristic will be used in the tree search and what the objective is. "
                                    )


        # A high-level description of the puzzle and what the heuristic should do.
        self.prompt_task = ("Act as a professional algorithm designer. "
                            "Design a heuristic to guide the tree search. "
                            "\n"
                            f"{self.prompt_func_context}"
                            )


        # The function’s required inputs
        self.prompt_func_inputs = ["state"]

        # The function’s outputs
        self.prompt_func_outputs = ["score"]

        # Explanation of inputs and outputs
        self.prompt_inout_inf = (
            "'state' is a list of 2D grids representing the puzzle after a move. "
            "The output named 'score' is the score for the puzzle state. "
        )

        # Additional constraints and clarifications
        self.prompt_other_inf = (
                                "Note that 'state' is a two levels nested list with integers in the second level sublist. "
                                 "'score' must be a an integer or float. "
                                 "Avoid utilizing the random component, and it is crucial to maintain self-consistency. "
                                 "Do not use libraries. "
                                 "Do not use 'while' loops. "
                                 "Do not give additional explanations. "
                                 "Don't create additional methods and please avoid nesting methods. "
                                 )

        # Examples of input and output
        self.prompt_example = (
        "'state' is represented by a two levels deep nested list."
        "The second level list represents a row of the puzzle as a list of integers. "
        "The first list index (index 0) is the top row of the puzzle. "
        "The highest list index is the bottom row of the puzzle. "
        "A 1 represents the tile numbered 1. "
        "A 20 represents the tile numbered 20. "
        "A 0 represents the empty slot. "
        "Each state has exactly one 0 (empty slot) somewhere in the puzzle. "
        "Only one tile can move into the empty space at any given time. "
        "Each integer represents a tile with its number. "
            "This heuristic should work for puzzles of any size, like 10x10 shown below.\n\n"
            "Goal configuration example:\n"
           "[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],\n"
            " [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],\n"
            " [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],\n"
            " [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],\n"
            " [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],\n"
            " [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],\n"
            " [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],\n"
            " [71, 72, 73, 74, 75, 76, 77, 78, 79, 80],\n"
            " [81, 82, 83, 84, 85, 86, 87, 88, 89, 90],\n"
            " [91, 92, 93, 94, 95, 96, 97, 98, 99, 0]]\n\n"

            "First example for 'state': "
            "[\n"
            " [[1, 13, 2, 14, 5, 6, 7, 8, 9, 10],\n"
            "  [21, 11, 4, 3, 15, 16, 17, 18, 19, 20],\n"
            "  [12, 22, 23, 24, 25, 26, 27, 28, 29, 30],\n"
            "  [32, 52, 33, 53, 34, 36, 37, 38, 39, 40],\n"
            "  [42, 41, 43, 44, 35, 45, 46, 48, 49, 50],\n"
            "  [61, 64, 31, 63, 55, 56, 47, 58, 59, 60],\n"
            "  [62, 72, 51, 54, 75, 65, 0, 68, 69, 70],\n"
            "  [71, 82, 73, 74, 77, 67, 57, 78, 79, 80],\n"
            "  [81, 83, 84, 85, 66, 76, 97, 87, 88, 90],\n"
            "  [91, 92, 93, 94, 95, 86, 96, 98, 89, 99]]\n"
            "]\n"

            "Second example for 'state': "
            "[\n"
            " [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],\n"
            "  [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],\n"
            "  [21, 22, 23, 24, 36, 25, 27, 28, 29, 30],\n"
            "  [31, 42, 32, 33, 34, 26, 37, 39, 49, 40],\n"
            "  [41, 52, 43, 44, 35, 46, 47, 38, 50, 60],\n"
            "  [61, 51, 53, 54, 45, 56, 57, 48, 80, 59],\n"
            "  [71, 62, 63, 64, 0, 55, 66, 58, 68, 78],\n"
            "  [81, 72, 73, 74, 65, 75, 67, 90, 69, 70],\n"
            "  [91, 83, 93, 84, 76, 86, 77, 87, 79, 99],\n"
            "  [92, 82, 94, 95, 85, 96, 97, 88, 98, 89]]\n"
            "]\n"
            "First example for 'score': "
            "6"
            "Second example for 'score': "
            "0.5"
        )

    def get_task(self):
        return self.prompt_task

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf

    def get_examples(self):
        return self.prompt_example


if __name__ == "__main__":
    getprompts = GetPrompts()
    print(getprompts.get_task())
