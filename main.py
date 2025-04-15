import argparse
import time
from mpi4py import MPI
from src.maze import create_maze
from src.explorer import Explorer

def run_explorer(width, height, maze_type):
    """Create maze and solve it, then return the total time taken and the number of moves."""
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
    parser.add_argument("--width", type=int, default=30, help="Width of the maze (default: 30)")
    parser.add_argument("--height", type=int, default=30, help="Height of the maze (default: 30)")
    args = parser.parse_args()

    # Setup MPI communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Run the maze explorer in parallel
    time_taken, moves = run_explorer(args.width, args.height, args.type)

    # Gather all results from the explorers
    all_results = comm.gather((rank, time_taken, moves), root=0)

    # Only root process compares and displays results
    if rank == 0:
        print("\n===============================")
        print("  Maze Exploration Results  ")
        print("===============================")

        best_rank, best_time, best_moves = all_results[0]
        for r, t, m in all_results:
            print(f"Explorer {r:2}: Time = {t:6.2f}s | Moves = {m:4}")
            if m < best_moves:  # Select the explorer with the least moves
                best_rank, best_time, best_moves = r, t, m

        print("\n-------------------------------")
        print(f"Best Explorer: Explorer {best_rank} solved the maze with {best_moves} moves in {best_time:.2f} seconds.")
        print("===============================")

if __name__ == "__main__":
    main()
