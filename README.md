## Question 1:
## Algorithm Utilized
The maze explorer uses the right-hand rule algorithm whereby the explorer keeps its right hand against the wall as it navigates the maze. It initially attempts to move to the right. If right is blocked, it moves forward, and if forward is blocked as well, it turns left. The explorer backtracks only when it has exhausted all possible moves. This strategy ensures that it will locate the exit if the maze is completely connected without isolated regions.

## Loop Detection
In order to detect and follow loops, the explorer remembers its last three moves in a deque. When the last three moves are the same, i.e., the explorer has encountered a loop, it begins backtracking. In testing, no loops were encountered in the static maze, which indicates the path followed was optimal and straight.

## Backtracking Strategy
Backtracking occurs when the explorer gets stuck or is unable to proceed. It returns to earlier positions and searches for other directions. The explorer returns to where there were other directions to attempt, which it then attempts. Backtracking was not necessary in the static maze test, however, since the right-hand rule had a simple and direct route to the end.

## Statistics Provided
The explorer prints the following performance metrics when it exits the maze:
- Total time elapsed
- Number of moves made
- Number of backtrack operations
- Average moves per second

### Example Output (Non-visualize):
```yaml
Total time elapsed: 0.00 seconds
Number of moves made: 1279
Number of backtrack operations: 0
Average moves per second: 1002900.51
```

### Example Output (Visualize):
```yaml
Total time elapsed: 42.36 seconds
Number of moves made: 1279
Number of backtrack operations: 0
Average moves per second: 30.19
```

Time difference is due to the visualization overhead, which adds delays for rendering the maze.

## Observations
I ran the automated explorer in two modes:
- Non-visual mode: `python main.py --type static`
- Visual mode: `python main.py --type static --visualize`

The explorer discovered the solution to the static maze in 1279 moves without going back. The visual mode was approximately 42 seconds as a function of the screen update time, and the non-visual mode took practically no time. The result suggests that performance difference is caused by rendering time in the visual mode and not inefficiencies in the algorithm.

## Summary
The maze explorer uses a simple but effective right-hand rule-based wall-following algorithm, an in-built loop detection mechanism, and backtracking strategy. It collects essential performance metrics, which offer insights to analyze the behavior of the explorer as a function of maze structure. The visual and non-visual mode output demonstrates the algorithm's efficiency, and performance difference shows the impact of graphics rendering.

## Question 2:
### Overview of main.py Code (With MPI4Py):
In the new main.py, we employed MPI4Py to run multiple explorers in parallel and solve the maze by automated exploration. We were interested in comparing the result of the explorers on the basis of time and the number of moves made.

Key steps:
1. MPI4Py Integration: MPI.COMM_WORLD allows parallel running, with multiple explorers solving the maze individually.
2. run_explorer() Function: This function is responsible for creating the maze, solving it, and timing it, utilizing time.perf_counter() for better accuracy.
3. Parallel Execution: Multiple explorers run in parallel, with each running the maze independently. The results are summed.
4. Results Collection: Time consumed and number of moves consumed are gathered from all the explorers and compared at the root process.
5. Best Explorer Selection: The best explorer is identified based on the least number of moves made and provided by the results.

### Summary of Results:
When running the program with the following command:
```
mpiexec -n 4 python main.py this is random (but for static maze the output is in question 3) 
```

```
pygame 2.6.1 (SDL 2.28.4, Python 3.12.2)
Hello from the pygame community. https://www.pygame.org/contribute.html
===============================
  Maze Exploration Results
===============================
Explorer  0: Time =   0.00s | Moves =  609
Explorer  1: Time =   0.00s | Moves =   83
Explorer  2: Time =   0.00s | Moves =   99
Explorer  3: Time =   0.00s | Moves =  414

-----------------------------
Best Explorer: Explorer 1 solved the maze with 83 moves in 0.00 seconds.
```

### Key Observations:
- Explorer 1 worked best as it solved the maze with 83 moves.
- Total time for all explorers was 0.00 seconds, which indicates that the solving of the maze was very fast (as would be expected in these smaller mazes).
- Move comparison functioned well in determining the best explorer, even though the time spent was minimal.

```python
import argparse
import time
from mpi4py import MPI
from src.maze import create_maze
from src.explorer import Explorer

def run_explorer(width, height, maze_type):
    "Create maze and solve it, then return the total time taken and the number of moves."
    maze = create_maze(width, height, maze_type)
    explorer = Explorer(maze, visualize=False)

    # Start the timer
    start_time = time.perf_counter()

    # Solve the maze and collect results
    time_taken, moves = explorer.solve()

    # End the timer
    end_time = time.perf_counter()

    # Calculate the total time taken for the exploration
    total_time = end_time - start_time

    return total_time, len(moves)

def main():
    parser = argparse.ArgumentParser(description="Maze Runner with MPI4Py")
    parser.add_argument("--type", choices=["random", "static"], default="random", help="Type of maze (random or static)")
    parser.add_argument("--width", type=int, default=30, help="Maze width (default: 30)")
    parser.add_argument("--height", type=int, default=30, help="Maze height (default: 30)")
    args = parser.parse_args()

    # Initialize MPI communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parallel run the maze explorer
    time_taken, moves = run_explorer(args.width, args.height, args.type)

    # All the results of explorers
    all_results = comm.gather((rank, time_taken, moves), root=0)

    # Comparing and displaying results are only done in the root process
    if rank == 0:
        print("===============================")
        print("  Maze Exploration Results  ")
        print("===============================")

        best_rank, best_time, best_moves = all_results[0]
        for r, t, m in all_results:
            print(f"Explorer {r:2}: Time = {t:6.2f}s | Moves = {m:4}")
            if m < best_moves:  # Select the explorer who has made the minimum moves
                best_rank, best_time, best_moves = r, t, m

        print("\n-------------------------------")
        print(f"Best Explorer: Explorer {best_rank} solved the maze with {best_moves} moves in {best_time:.2f} seconds.")
        print("==================================")

if __name__ == "__main__":
    main()
```

## Question 3:
### Observations and Analysis:
1. Time Taken:
   - All explorers solved the maze in 0.00 seconds, as expected given the smaller size of the static maze and the efficient running of the algorithm.
   - This can also be due to the very high precision of the timer, under which short values are not counted exactly.

2. Number of Moves:
   - All the adventurers performed exactly 1279 moves, which implies that the algorithm for solving the maze is consistent in all the parallel processes. Since the maze does not change, the number of moves is the same for all the adventurers.

3. Backtrack Operations:
   - Backtrack operations count 0 for each explorer, suggesting that the solution algorithm was good enough to never get stuck in dead ends.

### Key Insight:
- Consistency: Since the maze is not dynamic, outcomes between all explorers were identical with respect to moves and backtracking, ensuring that all explorers took the same path to solve the maze.
- Efficiency: The explorers efficiently solved the maze by making the same number of moves without any backtracking required.

### Output after use:
```
mpiexec -n 4 python main.py --type static
pygame 2.6.1 (SDL 2.28.4, Python 3.12.2)
Hello from the pygame community. https://www.pygame.org/contribute.html

=== Maze Exploration Statistics ===
Total time taken: 0.00 seconds
Total moves made: 1279
Number of backtrack operations: 0
Average moves per second: 657013.45
==================================

pygame 2.6.1 (SDL 2.28.4, Python 3.12.2)
Hello from the pygame community. https://www.pygame.org/contribute.html

=== Maze Exploration Statistics ===
Total time taken: 0.00 seconds
Total moves made: 1279
Number of backtrack operations: 0
Average moves per second: 630229.65
==================================

pygame 2.6.1 (SDL 2.28.4, Python 3.12.2)
Hello from the pygame community. https://www.pygame.org/contribute.html

=== Maze Exploration Statistics ===
Total time taken: 0.00 seconds
Total moves made: 1279
Number of backtrack operations: 0
Average moves per second: 686526.08
==================================

pygame 2.6.1 (SDL 2.28.4, Python 3.12.2)
Hello from the pygame community. https://www.pygame.org/contribute.html

=== Maze Exploration Statistics ===
Total time taken: 0.00 seconds
Total moves executed: 1279
Backtrack operations performed: 0
Average moves/second: 690591.51
==================================

===============================
  Maze Exploration Results
===============================
Explorer  0: Time =   0.00s | Moves = 1279
Explorer  1: Time =   0.00s | Moves = 1279
Explorer  2: Time =   0.00s | Moves = 1279
Explorer  3: Time =   0.00s | Moves = 1279

-------------------------------
Best Explorer: Explorer 0 solved the maze in 1279 moves in 0.00 seconds.
```

## Question 4:

## Identified Limitations of the Original Maze Explorer

The original version of the `Explorer` class used a **right-hand rule** algorithm with a custom backtracking mechanism. While functional, it exhibits several significant limitations:

1. **Inefficient Pathfinding**  
   The right-hand rule is a local, rule-based method that does not take into account the overall layout of the maze. It often explores irrelevant or longer paths before reaching the goal. This makes the solution inefficient, especially for large or open mazes.

2. **No Shortest Path Guarantee**  
   This approach does not guarantee that the path found is the shortest. It may eventually reach the goal, but the solution could be far from optimal in terms of the number of steps taken.

3. **Dependence on Backtracking**  
   The algorithm heavily relies on backtracking to recover from dead ends. This results in extra logic for tracking move history and manually finding previous fork points, which can slow down the solving process and increase complexity.

4. **Looping and Repetition Risk**  
   The explorer can revisit the same positions repeatedly, especially in mazes with cycles or open spaces. Although it uses a short move history to detect loops, this method is not robust and may fail to prevent infinite loops in certain configurations.

## Proposed Improvements

To address these limitations and improve the performance and reliability of the maze solver, the following enhancements were proposed:

### 1. **Implement Breadth-First Search (BFS)**
Replace the right-hand rule algorithm with **Breadth-First Search (BFS)**. BFS is a graph traversal algorithm that explores nodes level by level and guarantees the shortest path in unweighted mazes. Unlike the original algorithm, **BFS does not require any form of backtracking**, because it systematically expands the search frontier and never revisits nodes.

### 2. **Track Visited Nodes Using a Set**
Introduce a `visited` set to efficiently track which maze cells have already been explored. This ensures that the solver does not revisit the same cell and avoids redundant computation, making the traversal more efficient.

## Optional Improvements

Additional improvements that were considered (but not necessarily implemented in this iteration) include:

* **Separation of visualization and logic** for improved modularity and easier testing.
* **Tracking additional statistics**, such as total nodes explored or memory usage.

## Implemented Enhancements

The following two core enhancements were successfully implemented in the improved maze explorer:

### **Breadth-First Search (BFS) Integration**
* BFS explores all possible paths level by level, ensuring that the **shortest path** to the goal is found.
* Unlike the previous method, BFS does **not require backtracking** because it avoids dead ends by design.
* It naturally avoids loops and unnecessary movements, making it far more efficient and reliable.

### **Efficient Visited Node Tracking**
* A `visited` set was introduced to track which cells have already been processed.
* This ensures that nodes are never revisited, avoiding infinite loops and reducing the number of unnecessary steps.

## Outcome and Benefits

With these enhancements, the new maze explorer offers significant advantages over the original version:

* **Guaranteed shortest path** in all solvable mazes (unweighted).
* **No need for manual backtracking**, simplifying the logic and improving performance.
* **Faster execution** and reduced move count, especially in large or complex mazes.
* **Improved scalability** due to systematic and efficient traversal.

## Conclusion

By replacing the right-hand rule and backtracking logic with a robust **BFS-based algorithm**, the maze explorer has become significantly more efficient and intelligent. These improvements resolve the main limitations of the original design and provide a solid foundation for future extensions.

### Code for q4:
```python
import time
import pygame
from collections import deque
from typing import Tuple, List
from src.constants import BLUE, WHITE, CELL_SIZE, WINDOW_SIZE

class Explorer:
    def __init__(self, maze, visualize: bool = False):
        self.maze = maze
        self.x, self.y = maze.start_pos  # initial position
        self.moves = []
        self.start_time = None
        self.end_time = None
        self.visualize = visualize
        self.visited = set()  # To track visited positions
        self.backtrack_count = 0
        
        # Initialize visualization if needed
        if visualize:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            pygame.display.set_caption("Maze Explorer - BFS")
            self.clock = pygame.time.Clock()

    def can_move_to(self, x: int, y: int) -> bool:
        """Check whether a location is valid and available."""
        return 0 <= x < self.maze.width and 0 <= y < self.maze.height and self.maze.grid[y][x] == 0

    def bfs(self) -> List[Tuple[int, int]]:
        """Breadth-First Search for the shortest path."""
        queue = deque([(self.x, self.y, [])])  # (current_x, current_y, path_taken)
        self.visited.add((self.x, self.y))

        while queue:
            x, y, path = queue.popleft()

            # Check if we reached the goal
            if (x, y) == self.maze.end_pos:
                return path + [(x, y)]

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if self.can_move_to(nx, ny) and (nx, ny) not in self.visited:
                    queue.append((nx, ny, path + [(x, y)]))
                    self.visited.add((nx, ny))

        return []  # Return empty list if no solution

    def move(self, dx, dy):
        """Move the explorer."""
        self.x += dx
        self.y += dy
        self.moves.append((self.x, self.y))

    def solve(self) -> Tuple[float, List[Tuple[int, int]]]:
        """Solve the maze using BFS."""
        self.start_time = time.perf_counter()

        # Perform BFS to find the best path
        path = self.bfs()

        if not path:
            print("No solution found!")
            return 0.0, []  # No path found

        # Move along the path
        for x, y in path:
            self.move(x - self.x, y - self.y)

        self.end_time = time.perf_counter()
        time_taken = self.end_time - self.start_time

        if self.visualize:
            pygame.time.wait(2000)
            pygame.quit()

        self.print_statistics(time_taken)
        return time_taken, self.moves

    def print_statistics(self, time_taken: float):
        """Print maze-solving stats."""
        print("\n=== Maze Exploration Statistics ===")
        print(f"Total time taken: {time_taken:.2f} seconds")
        print(f"Total moves made: {len(self.moves)}")
        print(f"Average moves per second: {len(self.moves)/time_taken:.2f}")
        print("==================================\n")
```

### Output:
```
mpiexec -n 4 python main.py --type static
pygame 2.6.1 (SDL 2.28.4, Python 3.12.2)
Hello from the pygame community. https://www.pygame.org/contribute.html

=== Maze Exploration Statistics ===
Total time taken: 0.00 seconds
Total moves made: 128
Average moves per second: 59829.42
==================================

pygame 2.6.1 (SDL 2.28.4, Python 3.12.2)
Hello from the pygame community. https://www.pygame.org/contribute.html

=== Maze Exploration Statistics ===
Total time taken: 0.00 seconds
Total moves made: 128
Average moves per second: 54868.85
==================================

pygame 2.6.1 (SDL 2.28.4, Python 3.12.2)
Hello from the pygame community. https://www.pygame.org/contribute.html

=== Maze Exploration Statistics ===
Total time taken: 0.00 seconds
Total moves made: 128
Average moves per second: 55840.60
==================================

pygame 2.6.1 (SDL 2.28.4, Python 3.12.2)
Hello from the pygame community. https://www.pygame.org/contribute.html

=== Maze Exploration Statistics ===
Total time taken: 0.00 seconds
Total moves made: 128
Average moves per second: 52762.43
==================================

===============================
  Maze Exploration Results
===============================
Explorer  0: Time =   0.00s | Moves =  128
Explorer  1: Time =   0.00s | Moves =  128
Explorer 2: Time =   0.00s | Moves =  128
Explorer  3: Time =   0.00s | Moves =  128

-------------------------------
Best Explorer: Explorer 0 solved the maze in 0.00 seconds with 128 moves.

-------------------------------
(farahparallel) student@vg-DSAI-3202-20:~/Farah-Parallel-and-distributed-comp/maze-runner$ mpiexec -n 4 python main.py
pygame 2.6.1 (SDL 2.28.4, Python 3.12.2)
Hello from the pygame community. https://www.pygame.org/contribute.html

=== Maze Exploration Statistics ===
Total time taken: 0.00 seconds
Total moves made: 74
Average moves per second: 227807.92
==================================

pygame 2.6.1 (SDL 2.28.4, Python 3.12.2)
Hello from the pygame community. https://www.pygame.org/contribute.html

=== Maze Exploration Statistics ===
Total time taken: 0.00 seconds
Total moves made: 76
Average moves per second: 361076.53
==================================

pygame 2.6.1 (SDL 2.28.4, Python 3.12.2)
Hello from the pygame community. https://www.pygame.org/contribute.html

=== Maze Exploration Statistics ===
Total time taken: 0.00 seconds
Total moves made: 85
Average moves per second: 240924.69
==================================

pygame 2.6.1 (SDL 2.28.4, Python 3.12.2)
Hello from the pygame community. https://www.pygame.org/contribute.html

=== Maze Exploration Statistics ===
Total time taken: 0.00 seconds
Total moves made: 172
Average moves/sec: 471245.82
==================================

===============================
  Maze Exploration Results
===============================
Explorer  0: Time =   0.00s | Moves =  172
Explorer  1: Time =   0.00s | Moves =   74
Explorer  2: Time =   0.00s | Moves =   85
Explorer  3: Time =   0.00s | Moves =   76

-------------------------------
Best Explorer: Explorer 1 solved the maze in 74 moves in 0.00 seconds. (this is with random)
```

## Question 5: Performance Comparison and Analysis

### 1. Performance Comparison Results and Analysis
The performance of the enhanced BFS-based explorer was compared with the basic explorer that used the right-hand wall-following rule. Both were experimented on the same static maze configuration.

**Original Explorer (Right-Hand Rule):**
- Time taken: 0.00 seconds, approximately 42 seconds (when doing --visualize).
- Total moves made: 1279.
- Backtrack operations: 0 (backtracking logic implemented but not used in this test).
- Average moves per second: Over 650,000 in non-visual mode.
- Behavior: The explorer moved around walls, sometimes round and round, resulting in a path unnecessarily longer.

**Improved Explorer (BFS):**
- Elapsed time: 0.00 seconds (non-visual mode).
- Total moves made: 128.
- Backtrack operations: Not applicable (BFS avoids unnecessary backtracking by design).
- Average moves per second: Around 55,000.
- Behavior: The explorer estimated the best course from start to finish and cut it straight down the middle without retracing. 

**Observations:**
- The enhanced explorer finished the maze with only 128 moves, an impressive reduction from the 1279 of the original.
- BFS was also much quicker, reducing moves by over 90%.
- The simple method used a lot of steps navigating dead ends or up walls.
- There were virtually no differences in time in non-visual mode, 
- Both adventurers completed the maze successfully, but BFS was clearly more efficient.

### 3. Visuals
I had an issue with visualizing the maze on my device im sorryyyyy

### 3. Trade-offs and New Limitations Introduced
While the implementation of BFS is more efficient and easier than the original, it also creates a couple of trade-offs:

**Advantages of BFS:**
- Guarantees the shortest path.
- No loops or redundant moves.
- Simpler, more readable code.
- Scales well for larger and more complex mazes.

**Limitations of BFS:**
- Requires full knowledge of the maze before solving. It won't work in real-world situations where the environment is not fully known.
- Uses more memory due to the need to track visited nodes and store paths in a queue.
- Doesn't simulate human-like or real-world exploration behavior, which might be useful for certain types of visualizations or educational demos.
- Less exploratory than the original because the entire path is found and then walked, rather than found step by step.

### Summary
The enhanced explorer with Breadth-First Search significantly enhances maze-solving efficiency over the original right-hand rule. It locates the goal in much fewer moves and performs better in automated environments where the maze layout is already known in advance.

Although it sacrifices some of the human-like and realistic exploration behavior, the BFS algorithm is much more practical to use in optimized automated solving and is highly recommended for academic or performance situations.

## Question 6:
I scored 128 moves, and also a solution with no backtracking since I used BFS! 
