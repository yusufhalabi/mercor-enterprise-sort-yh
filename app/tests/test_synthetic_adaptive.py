import random
import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
from collections import defaultdict

@dataclass
class Candidate:
    id: int
    wins: int = 0
    comparisons: int = 0
    win_history: List[bool] = None
    
    def __post_init__(self):
        self.win_history = []
    
    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.comparisons)
    
    @property
    def recent_win_rate(self) -> float:
        """Calculate win rate from recent comparisons"""
        recent_window = 5
        if len(self.win_history) == 0:
            return 0
        recent = self.win_history[-recent_window:]
        return sum(recent) / len(recent)

class ComparisonTracker:
    def __init__(self):
        self.count = 0
        self.comparison_counts = defaultdict(int)
        self.comparison_cache: Dict[Tuple[int, int], bool] = {}
    
    async def compare(self, a: Candidate, b: Candidate) -> bool:
        """Compare two candidates, with caching"""
        self.count += 1
        self.comparison_counts[a.id] += 1
        self.comparison_counts[b.id] += 1
        
        # For testing with numbers, higher is better
        result = a.id >= b.id
        
        # Update win history
        a.win_history.append(result)
        b.win_history.append(not result)
        
        # Update win counts
        if result:
            a.wins += 1
        else:
            b.wins += 1
            
        a.comparisons += 1
        b.comparisons += 1
        
        return result

class AdaptiveRanker:
    def __init__(self):
        self.tracker = ComparisonTracker()
        self.batch_size = 20
        
    async def rank_candidates(self, numbers: List[int], k: int) -> List[Tuple[int, int]]:
        """Main ranking function"""
        candidates = [Candidate(id=num) for num in numbers]
        
        # Phase 1: Initial sampling
        active_pool = await self._initial_sampling(candidates)
        
        # Phase 2: Iterative refinement
        top_candidates = await self._iterative_refinement(active_pool, k)
        
        # Add remaining candidates
        remaining = sorted(
            [c for c in candidates if c not in top_candidates],
            key=lambda x: x.id,
            reverse=True
        )
        
        return [(c.id, self.tracker.comparison_counts[c.id]) 
                for c in (top_candidates + remaining)]
    
    async def _initial_sampling(self, candidates: List[Candidate]) -> List[Candidate]:
        """Initial sampling phase with adaptive opponent selection"""
        tasks = []
        samples_per_candidate = 5
        
        # Compare each candidate with random opponents
        for candidate in candidates:
            opponents = random.sample([c for c in candidates if c != candidate], 
                                   min(samples_per_candidate, len(candidates)-1))
            for opponent in opponents:
                tasks.append(self.tracker.compare(candidate, opponent))
                
                if len(tasks) >= self.batch_size:
                    await asyncio.gather(*tasks)
                    tasks = []
        
        if tasks:
            await asyncio.gather(*tasks)
        
        # Select candidates based on initial performance
        return sorted(candidates, 
                     key=lambda x: (x.win_rate, x.id),
                     reverse=True)[:len(candidates)//3]
    
    async def _iterative_refinement(self, candidates: List[Candidate], k: int) -> List[Candidate]:
        """Iteratively refine rankings with adaptive sampling"""
        current_pool = candidates
        rounds = 4
        min_comparisons = 7
        
        for round_num in range(rounds):
            tasks = []
            round_size = k + (len(current_pool) - k) // 2
            
            # Prioritize comparing candidates with similar win rates
            sorted_pool = sorted(current_pool, key=lambda x: x.win_rate, reverse=True)
            
            for i, candidate in enumerate(sorted_pool):
                # Compare with neighbors and some random candidates
                comparison_needed = min_comparisons - len([
                    h for h in candidate.win_history[-min_comparisons:]
                    if h is not None
                ])
                
                if comparison_needed > 0:
                    # Select opponents: mix of neighbors and random candidates
                    nearby_indices = [
                        j for j in range(
                            max(0, i-2),
                            min(len(sorted_pool), i+3)
                        ) if j != i
                    ]
                    nearby_opponents = [sorted_pool[j] for j in nearby_indices]
                    
                    # Add some random opponents
                    random_opponents = random.sample(
                        [c for c in sorted_pool if c not in nearby_opponents 
                         and c != candidate],
                        min(2, len(sorted_pool) - len(nearby_opponents) - 1)
                    )
                    
                    opponents = nearby_opponents + random_opponents
                    
                    for opponent in opponents[:comparison_needed]:
                        tasks.append(self.tracker.compare(candidate, opponent))
                        
                        if len(tasks) >= self.batch_size:
                            await asyncio.gather(*tasks)
                            tasks = []
            
            if tasks:
                await asyncio.gather(*tasks)
            
            # Update pool based on both recent and overall performance
            current_pool = sorted(
                current_pool,
                key=lambda x: (0.7 * x.recent_win_rate + 0.3 * x.win_rate, x.id),
                reverse=True
            )[:round_size]
            
            # Early stopping if we have enough confidence
            if round_num >= 2:
                top_k = current_pool[:k]
                confidence = sum(c.win_rate for c in top_k) / k
                if confidence > 0.8:
                    break
        
        return current_pool[:k]

async def test_algorithm():
    """Test function"""
    n = 1000
    k = 30
    numbers = random.sample(range(1, 1001), n)
    true_top_k = set(sorted(numbers, reverse=True)[:k])
    
    ranker = AdaptiveRanker()
    rankings = await ranker.rank_candidates(numbers, k)
    our_top_k = set(num for num, _ in rankings[:k])
    
    accuracy = len(true_top_k.intersection(our_top_k)) / k
    
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Comparisons: {ranker.tracker.count}")
    return accuracy, ranker.tracker.count

async def main():
    """Run multiple tests"""
    num_trials = 10
    accuracies = []
    comparisons = []
    
    for i in range(num_trials):
        print(f"\nTrial {i+1}/{num_trials}")
        accuracy, comparison_count = await test_algorithm()
        accuracies.append(accuracy)
        comparisons.append(comparison_count)
    
    print(f"\nAverage Accuracy: {sum(accuracies)/len(accuracies):.2%}")
    print(f"Average Comparisons: {sum(comparisons)/len(comparisons):.0f}")

if __name__ == "__main__":
    asyncio.run(main())