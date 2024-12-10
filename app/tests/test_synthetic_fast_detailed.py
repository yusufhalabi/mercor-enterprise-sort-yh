import random
import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import defaultdict

@dataclass
class Candidate:
    id: int
    wins: int = 0
    comparisons: int = 0
    
    @property
    def confidence_score(self) -> float:
        """Score that combines win rate with comparison confidence"""
        comparison_confidence = min(1.0, self.comparisons / 3)
        win_rate = self.wins / max(1, self.comparisons)
        return win_rate * comparison_confidence

class ComparisonTracker:
    def __init__(self):
        self.count = 0
        self.comparison_counts = defaultdict(int)
        self.comparison_cache: Dict[Tuple[int, int], bool] = {}
    
    async def compare(self, a: Candidate, b: Candidate) -> bool:
        """Compare two candidates and cache the result"""
        cache_key = (a.id, b.id)
        if cache_key in self.comparison_cache:
            return self.comparison_cache[cache_key]
            
        self.count += 1
        self.comparison_counts[a.id] += 1
        self.comparison_counts[b.id] += 1
        
        result = a.id >= b.id
        self.comparison_cache[cache_key] = result
        self.comparison_cache[(b.id, a.id)] = not result
        
        if result:
            a.wins += 1
        else:
            b.wins += 1
            
        a.comparisons += 1
        b.comparisons += 1
        
        return result

class ImprovedRanker:
    def __init__(self):
        self.tracker = ComparisonTracker()
        self.batch_size = 20
        
    async def rank_candidates(self, numbers: List[int], k: int) -> List[Tuple[int, int]]:
        candidates = [Candidate(id=num) for num in numbers]
        
        # Phase 1: Initial balanced sampling
        await self._initial_sampling(candidates)
        
        # Phase 2: Select and refine potential top candidates
        potential_top = self._select_potential_top(candidates, k)
        final_top = await self._refine_top(potential_top, k)
        
        # Handle remaining candidates
        remaining = sorted(
            [c for c in candidates if c not in final_top],
            key=lambda x: x.id,
            reverse=True
        )
        
        return [(c.id, self.tracker.comparison_counts[c.id]) 
                for c in (final_top + remaining)]

    async def _initial_sampling(self, candidates: List[Candidate]) -> None:
        """Ensure each candidate gets at least one comparison"""
        pairs = []
        remaining = candidates.copy()
        while remaining:
            if len(remaining) >= 2:
                a, b = random.sample(remaining, 2)
                pairs.append((a, b))
                remaining.remove(a)
                remaining.remove(b)
            elif remaining:
                a = remaining[0]
                b = random.choice([c for c in candidates if c != a])
                pairs.append((a, b))
                remaining = []
        
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            await asyncio.gather(*(self.tracker.compare(a, b) for a, b in batch))

    def _select_potential_top(self, candidates: List[Candidate], k: int) -> List[Candidate]:
        """Select potential top candidates based on initial sampling"""
        return sorted(
            candidates,
            key=lambda x: (x.confidence_score, x.id),
            reverse=True
        )[:k * 2]

    async def _refine_top(self, candidates: List[Candidate], k: int) -> List[Candidate]:
        """Refined comparison of top candidates"""
        tasks = []
        
        for i, candidate in enumerate(candidates):
            if i > 0:
                tasks.append(self.tracker.compare(candidate, candidates[i-1]))
            
            if i < len(candidates) - 1:
                tasks.append(self.tracker.compare(candidate, candidates[i+1]))
            
            others = [c for c in candidates if c != candidate and 
                     abs(candidates.index(c) - i) > 1]
            if others:
                tasks.append(self.tracker.compare(candidate, random.choice(others)))
            
            if len(tasks) >= self.batch_size:
                await asyncio.gather(*tasks)
                tasks = []
        
        if tasks:
            await asyncio.gather(*tasks)
        
        return sorted(
            candidates,
            key=lambda x: (x.confidence_score, x.id),
            reverse=True
        )[:k]