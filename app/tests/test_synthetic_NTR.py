import random
import asyncio
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import csv
import json
import matplotlib.pyplot as plt
import statistics

@dataclass
class Candidate:
    id: int
    wins: int = 0
    losses: int = 0
    comparisons: int = 0
    recent_wins: int = 0  # Track recent performance
    recent_comparisons: int = 0

class ComparisonTracker:
    def __init__(self):
        self.count = 0
        self.comparison_counts = defaultdict(int)
    
    async def pairwise_compare(self, a: Candidate, b: Candidate) -> bool:
        self.count += 1
        self.comparison_counts[a.id] += 1
        self.comparison_counts[b.id] += 1
        return a.id >= b.id

class OptimizedRanker:
    def __init__(self):
        self.tracker = ComparisonTracker()
        self.comparison_cache = {}
        self.batch_size = 20
        
    async def rank_candidates(self, numbers: List[int], k: int) -> List[Tuple[int, int]]:
        candidates = [Candidate(id=num) for num in numbers]
        
        # Phase 1: Broad sampling to identify potential top candidates
        potential_top = await self._broad_sampling(candidates, k * 3)
        
        # Phase 2: Focused refinement with multiple rounds
        final_top = await self._iterative_refinement(potential_top, k)
        
        # Handle remaining candidates
        remaining = [c for c in candidates if c not in final_top]
        
        return [(c.id, self.tracker.comparison_counts[c.id]) for c in (final_top + remaining)]

    async def _broad_sampling(self, candidates: List[Candidate], target_size: int) -> List[Candidate]:
        """Initial broad sampling to identify potential top candidates"""
        # Compare each candidate against multiple random opponents
        comparison_tasks = []
        num_samples = 10  # Number of random comparisons per candidate
        
        for candidate in candidates:
            opponents = random.sample([c for c in candidates if c != candidate], 
                                   min(num_samples, len(candidates)-1))
            for opponent in opponents:
                comparison_tasks.append(self._compare_pair(candidate, opponent))
                
            if len(comparison_tasks) >= self.batch_size:
                await asyncio.gather(*comparison_tasks)
                comparison_tasks = []
        
        if comparison_tasks:
            await asyncio.gather(*comparison_tasks)
        
        # Select candidates based on win ratio
        scored_candidates = [
            (c, c.wins / max(1, c.comparisons))
            for c in candidates
        ]
        
        return [c for c, _ in sorted(scored_candidates, 
                                   key=lambda x: x[1], 
                                   reverse=True)][:target_size]

    async def _iterative_refinement(self, candidates: List[Candidate], k: int) -> List[Candidate]:
        """Iteratively refine rankings with multiple rounds"""
        current_candidates = candidates
        rounds = 3  # Number of refinement rounds
        
        for round_num in range(rounds):
            # Reset recent performance tracking
            for c in current_candidates:
                c.recent_wins = 0
                c.recent_comparisons = 0
            
            # Perform round-robin comparisons in current group
            comparison_tasks = []
            for i, cand_a in enumerate(current_candidates):
                # Compare with more opponents in earlier rounds
                num_opponents = max(5, len(current_candidates) // (round_num + 2))
                opponents = random.sample(
                    [c for c in current_candidates if c != cand_a],
                    min(num_opponents, len(current_candidates)-1)
                )
                
                for cand_b in opponents:
                    comparison_tasks.append(self._compare_pair_with_tracking(cand_a, cand_b))
                    
                if len(comparison_tasks) >= self.batch_size:
                    await asyncio.gather(*comparison_tasks)
                    comparison_tasks = []
            
            if comparison_tasks:
                await asyncio.gather(*comparison_tasks)
            
            # Update candidate pool based on recent performance
            keep_size = k + (len(current_candidates) - k) // 2
            current_candidates = sorted(
                current_candidates,
                key=lambda x: (
                    x.recent_wins / max(1, x.recent_comparisons),  # Recent performance
                    x.wins / max(1, x.comparisons)  # Overall performance as tiebreaker
                ),
                reverse=True
            )[:keep_size]
        
        return current_candidates[:k]

    async def _compare_pair_with_tracking(self, a: Candidate, b: Candidate):
        """Compare a pair and track both overall and recent performance"""
        result = await self._get_comparison_result(a, b)
        
        if result:
            a.wins += 1
            a.recent_wins += 1
            b.losses += 1
        else:
            b.wins += 1
            b.recent_wins += 1
            a.losses += 1
        
        a.comparisons += 1
        b.comparisons += 1
        a.recent_comparisons += 1
        b.recent_comparisons += 1

    async def _get_comparison_result(self, a: Candidate, b: Candidate) -> bool:
        """Get cached or new comparison result"""
        cache_key = (a.id, b.id)
        if cache_key in self.comparison_cache:
            return self.comparison_cache[cache_key]
        
        result = await self.tracker.pairwise_compare(a, b)
        self.comparison_cache[cache_key] = result
        self.comparison_cache[(b.id, a.id)] = not result
        return result

    async def _compare_pair(self, a: Candidate, b: Candidate):
        """Basic pair comparison without recent performance tracking"""
        result = await self._get_comparison_result(a, b)
        
        if result:
            a.wins += 1
            b.losses += 1
        else:
            b.wins += 1
            a.losses += 1
        
        a.comparisons += 1
        b.comparisons += 1

# Rest of the code (BenchmarkSuite, main function, etc.) remains the same

@dataclass
class BenchmarkResult:
    accuracy: float
    comparison_count: int
    execution_time: float
    top_10_accuracy: float
    top_20_accuracy: float
    perfect_positions: int

class BenchmarkSuite:
    def __init__(self, num_trials=10):
        self.num_trials = num_trials
        self.k = 30
        self.n = 1000
    
    def calculate_accuracy(self, rankings: List[Tuple[int, int]], numbers: List[int], k: int) -> float:
        true_top_k = set(sorted(numbers, reverse=True)[:k])
        our_top_k = set(num for num, _ in rankings[:k])
        return len(true_top_k.intersection(our_top_k)) / k

    def count_perfect_positions(self, ranked: List[int], true: List[int]) -> int:
        return sum(1 for r, t in zip(ranked, true) if r == t)

    async def run_single_trial(self) -> BenchmarkResult:
        numbers = random.sample(range(1, 1001), self.n)
        ranker = OptimizedRanker()
        
        start_time = time.time()
        rankings = await ranker.rank_candidates(numbers, self.k)
        execution_time = time.time() - start_time
        
        ranked_numbers = [num for num, _ in rankings]
        true_order = sorted(numbers, reverse=True)
        
        return BenchmarkResult(
            accuracy=self.calculate_accuracy(rankings, numbers, self.k),
            comparison_count=ranker.tracker.count,
            execution_time=execution_time,
            top_10_accuracy=self.calculate_accuracy(rankings, numbers, 10),
            top_20_accuracy=self.calculate_accuracy(rankings, numbers, 20),
            perfect_positions=self.count_perfect_positions(ranked_numbers[:self.k], true_order[:self.k])
        )

    def save_results(self, results: List[BenchmarkResult], filename: str):
        Path("benchmark_results").mkdir(exist_ok=True)
        
        # Save detailed results
        report = {
            "accuracy": {
                "mean": statistics.mean(r.accuracy for r in results),
                "std": statistics.stdev(r.accuracy for r in results),
                "min": min(r.accuracy for r in results),
                "max": max(r.accuracy for r in results)
            },
            "comparisons": {
                "mean": statistics.mean(r.comparison_count for r in results),
                "std": statistics.stdev(r.comparison_count for r in results),
                "min": min(r.comparison_count for r in results),
                "max": max(r.comparison_count for r in results)
            },
            "execution_time": {
                "mean": statistics.mean(r.execution_time for r in results),
                "std": statistics.stdev(r.execution_time for r in results)
            },
            "top_10_accuracy": {
                "mean": statistics.mean(r.top_10_accuracy for r in results),
                "std": statistics.stdev(r.top_10_accuracy for r in results)
            },
            "top_20_accuracy": {
                "mean": statistics.mean(r.top_20_accuracy for r in results),
                "std": statistics.stdev(r.top_20_accuracy for r in results)
            }
        }
        
        with open(f"benchmark_results/{filename}.json", 'w') as f:
            json.dump(report, f, indent=2)
            
        return report

    def create_visualizations(self, results: List[BenchmarkResult]):
        # Accuracy vs Comparisons
        plt.figure(figsize=(10, 6))
        plt.scatter([r.comparison_count for r in results], 
                   [r.accuracy for r in results],
                   alpha=0.6)
        plt.xlabel('Number of Comparisons')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Number of Comparisons')
        plt.savefig('benchmark_results/accuracy_vs_comparisons.png')
        plt.close()
        
        # Performance metrics
        metrics = ['Top 10', 'Top 20', 'Top 30']
        means = [
            statistics.mean(r.top_10_accuracy for r in results),
            statistics.mean(r.top_20_accuracy for r in results),
            statistics.mean(r.accuracy for r in results)
        ]
        
        plt.figure(figsize=(10, 6))
        plt.bar(metrics, means)
        plt.ylabel('Accuracy')
        plt.title('Accuracy at Different K Values')
        plt.savefig('benchmark_results/accuracy_by_k.png')
        plt.close()

async def main():
    # Run benchmarks
    benchmark = BenchmarkSuite(num_trials=10)
    results = []
    
    print("Running benchmarks...")
    for i in range(benchmark.num_trials):
        print(f"Trial {i + 1}/{benchmark.num_trials}")
        result = await benchmark.run_single_trial()
        results.append(result)
        print(f"Accuracy: {result.accuracy:.2%}, Comparisons: {result.comparison_count}")
    
    # Save and display results
    report = benchmark.save_results(results, "benchmark_results")
    benchmark.create_visualizations(results)
    
    print("\nFinal Results:")
    print(f"Average Accuracy: {report['accuracy']['mean']:.2%}")
    print(f"Average Comparisons: {report['comparisons']['mean']:.0f}")
    print(f"Average Execution Time: {report['execution_time']['mean']:.3f}s")
    print(f"Top 10 Accuracy: {report['top_10_accuracy']['mean']:.2%}")
    print(f"Top 20 Accuracy: {report['top_20_accuracy']['mean']:.2%}")
    
    # Save example input/output for verification
    n = 1000
    k = 30
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    numbers = random.sample(range(1, 1001), n)
    
    ranker = OptimizedRanker()
    rankings = await ranker.rank_candidates(numbers, k)
    
    # Save input
    Path("results").mkdir(exist_ok=True)
    with open(Path("results") / f"input_{timestamp}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Number'])
        for idx, num in enumerate(numbers):
            writer.writerow([idx + 1, num])
    
    # Save output
    accuracy = benchmark.calculate_accuracy(rankings, numbers, k)
    save_results_to_csv(rankings, ranker.tracker.count, accuracy, f"output_{timestamp}.csv")

def save_results_to_csv(rankings: List[Tuple[int, int]], 
                       total_comparisons: int,
                       accuracy: float,
                       filename: str):
    Path("results").mkdir(exist_ok=True)
    filepath = Path("results") / filename
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Number', 'Win_Count'])
        
        for rank, (number, wins) in enumerate(rankings, 1):
            writer.writerow([rank, number, wins])
            
        writer.writerow([])
        writer.writerow(['Total Comparisons:', total_comparisons])
        writer.writerow(['Top K Accuracy:', f"{accuracy:.2%}"])

if __name__ == "__main__":
    asyncio.run(main())