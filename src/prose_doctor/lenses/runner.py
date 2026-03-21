"""LensRunner — executes lenses in dependency order."""
from __future__ import annotations

from prose_doctor.lenses import Lens, LensResult, LensRegistry
from prose_doctor.providers import ProviderPool


class LensRunner:
    """Executes lenses in topological order, passing consumed results to meta-lenses."""

    def __init__(
        self,
        registry: LensRegistry,
        providers: ProviderPool,
        tier_filter: str | None = None,
        tiers: dict[str, str] | None = None,
    ) -> None:
        self._registry = registry
        self._providers = providers
        self._tier_filter = tier_filter
        self._tiers = tiers or {}

    def _eligible_lenses(self) -> list[Lens]:
        if not self._tier_filter:
            return self._registry.all_lenses()
        tier_rank = {"experimental": 0, "validated": 1, "stable": 2}
        min_rank = tier_rank.get(self._tier_filter, 2)
        eligible = []
        for lens in self._registry.all_lenses():
            lens_tier = self._tiers.get(lens.name, "experimental")
            if tier_rank.get(lens_tier, 0) >= min_rank:
                eligible.append(lens)
        return eligible

    def _toposort(self, lenses: list[Lens]) -> list[Lens]:
        by_name = {l.name: l for l in lenses}
        visited: set[str] = set()
        in_stack: set[str] = set()
        order: list[Lens] = []

        def visit(name: str) -> None:
            if name in in_stack:
                raise ValueError(f"Dependency cycle detected involving '{name}'")
            if name in visited:
                return
            in_stack.add(name)
            lens = by_name.get(name)
            if lens:
                for dep in lens.consumes_lenses:
                    if dep in by_name:
                        visit(dep)
            in_stack.remove(name)
            visited.add(name)
            if lens:
                order.append(lens)

        for lens in lenses:
            visit(lens.name)
        return order

    def run_all(self, text: str, filename: str) -> dict[str, LensResult]:
        lenses = self._eligible_lenses()
        sorted_lenses = self._toposort(lenses)
        results: dict[str, LensResult] = {}
        for lens in sorted_lenses:
            consumed = None
            if lens.consumes_lenses:
                consumed = {dep: results[dep] for dep in lens.consumes_lenses if dep in results}
            results[lens.name] = lens.analyze(text, filename, self._providers, consumed)
        return results

    def run_one(self, lens_name: str, text: str, filename: str) -> LensResult:
        lens = self._registry.get(lens_name)
        if lens is None:
            raise KeyError(f"Unknown lens: {lens_name}")
        needed: set[str] = set()

        def collect(name: str) -> None:
            l = self._registry.get(name)
            if l:
                for dep in l.consumes_lenses:
                    if dep not in needed:
                        needed.add(dep)
                        collect(dep)

        collect(lens_name)
        all_lenses = [self._registry.get(n) for n in needed if self._registry.get(n)]
        all_lenses.append(lens)
        sorted_lenses = self._toposort(all_lenses)
        results: dict[str, LensResult] = {}
        for l in sorted_lenses:
            consumed = None
            if l.consumes_lenses:
                consumed = {d: results[d] for d in l.consumes_lenses if d in results}
            results[l.name] = l.analyze(text, filename, self._providers, consumed)
        return results[lens_name]
