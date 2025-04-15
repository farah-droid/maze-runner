import time
import pygame
from collections import deque
from typing import Tuple, List
from src.constants import BLUE, WHITE, CELL_SIZE, WINDOW_SIZE

class Explorer:
    def __init__(self, maze, visualize: bool = True):
        self.maze = maze
        self.x, self.y = maze.start_pos  # Starting position
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
        """Check if a position is valid and open."""
        return 0 <= x < self.maze.width and 0 <= y < self.maze.height and self.maze.grid[y][x] == 0

    def bfs(self) -> List[Tuple[int, int]]:
        """Breadth-First Search to find the shortest path."""
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

        return []  # Return an empty list if no solution

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
        for (x, y) in path:
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
