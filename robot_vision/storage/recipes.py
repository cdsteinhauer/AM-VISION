from __future__ import annotations

import json
import re
from pathlib import Path

from robot_vision.inspection.models import InspectionRecipe


class RecipeStore:
    def __init__(self, recipe_dir: Path):
        self.recipe_dir = recipe_dir
        self.recipe_dir.mkdir(parents=True, exist_ok=True)

    def list_names(self) -> list[str]:
        names = sorted(path.stem for path in self.recipe_dir.glob("*.json"))
        if "default" not in names:
            self.save(InspectionRecipe.default())
            names.insert(0, "default")
        return names

    def load(self, name: str = "default") -> InspectionRecipe:
        path = self._path(name)
        if not path.exists():
            if name == "default":
                recipe = InspectionRecipe.default()
                self.save(recipe)
                return recipe
            raise FileNotFoundError(f"Recipe not found: {name}")
        with path.open("r", encoding="utf-8") as handle:
            return InspectionRecipe.from_dict(json.load(handle))

    def save(self, recipe: InspectionRecipe) -> Path:
        path = self._path(recipe.name)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(recipe.to_dict(), handle, indent=2)
        return path

    def delete(self, name: str) -> Path:
        path = self._path(name)
        if not path.exists():
            raise FileNotFoundError(f"Recipe not found: {name}")
        path.unlink()
        return path

    def _path(self, name: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip() or "default")
        return self.recipe_dir / f"{safe}.json"
