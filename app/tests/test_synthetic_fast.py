import random
import asyncio
from typing import List, Tuple
from datetime import datetime
from app.tests.test_synthetic_fast_detailed import ImprovedRanker

async def run_single_test(n: int = 1000, k: int = 30) -> Tuple[List[int], List[Tuple[int, int]], dict]:
    """Run a single test and return results with accuracies at different depths"""
    numbers = random.sample(range(1, n+1), n)
    true_ranking = sorted(numbers, reverse=True)
    
    ranker = ImprovedRanker()
    rankings = await ranker.rank_candidates(numbers, k)
    ranked_numbers = [num for num, _ in rankings]
    
    # Calculate accuracy at different depths
    check_depths = [10, 30, 50, 100, 500, 1000]
    accuracies = {}
    
    for depth in check_depths:
        if depth > len(true_ranking):
            continue
            
        # Get the set of numbers that should be in this range
        true_set = set(true_ranking[:depth])
        our_set = set(ranked_numbers[:depth])
        
        # Calculate accuracy as the proportion of correct numbers in this range
        accuracy = len(true_set.intersection(our_set)) / depth
        accuracies[depth] = accuracy
    
    return numbers, rankings, accuracies, ranker.tracker.count

async def main():
    trials = 5  # Number of trials
    
    print("\nAccuracy Results (N=1000, K=30):")
    print("-" * 75)
    print(f"{'Depth':>8} | {'Set Accuracy':>12} | {'Std Dev':>10} | {'Expected in Range':>15}")
    print("-" * 75)
    
    # Run multiple trials and collect results
    all_accuracies = {10: [], 30: [], 50: [], 100: [], 500: [], 1000: []}
    
    for _ in range(trials):
        _, _, accuracies, _ = await run_single_test()
        for depth, acc in accuracies.items():
            all_accuracies[depth].append(acc)
    
    # Calculate and display average accuracies
    for depth in [10, 30, 50, 100, 500, 1000]:
        if depth in all_accuracies and all_accuracies[depth]:
            accs = all_accuracies[depth]
            avg_acc = sum(accs) / len(accs)
            std_dev = (sum((x - avg_acc) ** 2 for x in accs) / len(accs)) ** 0.5
            expected = depth
            print(f"{depth:8d} | {avg_acc:11.2%} | {std_dev:9.2%} | {expected:15d}")

if __name__ == "__main__":
    asyncio.run(main())