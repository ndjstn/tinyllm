"""Prompt loader for TinyLLM.

This module provides the PromptLoader class for loading and caching
prompt definitions from YAML files.
"""

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from tinyllm.prompts.schema import PromptDefinition


class PromptLoader:
    """Loads and caches prompt definitions from YAML files.

    The PromptLoader maps prompt IDs to file paths and caches loaded
    prompts for efficient reuse.
    """

    # Category prefix to folder name mapping
    CATEGORY_FOLDERS = {
        "router": "routing",
        "specialist": "specialists",
        "thinking": "thinking",
        "tool": "tools",
        "grading": "grading",
        "meta": "meta",
        "memory": "memory",
    }

    def __init__(self, prompts_dir: Optional[Path | str] = None):
        """Initialize prompt loader.

        Args:
            prompts_dir: Root directory containing prompt YAML files.
                         Defaults to 'prompts' in the project root.
        """
        if prompts_dir is None:
            # Default to prompts/ in the project root
            prompts_dir = Path(__file__).parent.parent.parent.parent / "prompts"
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, PromptDefinition] = {}

    def load(self, prompt_id: str) -> PromptDefinition:
        """Load a prompt by ID.

        Prompt IDs follow the pattern: category.name.version
        E.g., "router.task_classifier.v1" loads from
        prompts/routing/task_classifier.yaml

        Args:
            prompt_id: Unique prompt identifier.

        Returns:
            Loaded prompt definition.

        Raises:
            FileNotFoundError: If prompt file doesn't exist.
            ValueError: If prompt YAML is invalid.
        """
        # Return cached if available
        if prompt_id in self._cache:
            return self._cache[prompt_id]

        # Parse ID to path
        path = self._id_to_path(prompt_id)

        if not path.exists():
            raise FileNotFoundError(
                f"Prompt not found: {prompt_id} (expected at {path})"
            )

        # Load YAML
        with open(path) as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty prompt file: {path}")

        # Validate and cache
        prompt = PromptDefinition(**data)
        self._cache[prompt_id] = prompt
        return prompt

    def load_by_path(self, path: Path | str) -> PromptDefinition:
        """Load a prompt directly from a file path.

        Args:
            path: Path to the prompt YAML file.

        Returns:
            Loaded prompt definition.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If YAML is invalid.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty prompt file: {path}")

        prompt = PromptDefinition(**data)
        self._cache[prompt.id] = prompt
        return prompt

    def list_prompts(self, category: Optional[str] = None) -> List[str]:
        """List available prompt IDs.

        Args:
            category: Optional category to filter by.

        Returns:
            List of prompt IDs.
        """
        prompt_ids = []

        if category:
            folder = self.CATEGORY_FOLDERS.get(category, category)
            search_dir = self.prompts_dir / folder
            if search_dir.exists():
                for path in search_dir.glob("*.yaml"):
                    prompt_ids.append(self._path_to_id(path, folder))
        else:
            for folder in self.prompts_dir.iterdir():
                if folder.is_dir() and not folder.name.startswith("."):
                    for path in folder.glob("*.yaml"):
                        prompt_ids.append(self._path_to_id(path, folder.name))

        return sorted(prompt_ids)

    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self._cache.clear()

    def _id_to_path(self, prompt_id: str) -> Path:
        """Convert prompt ID to file path.

        Args:
            prompt_id: Prompt identifier.

        Returns:
            Path to the prompt file.
        """
        parts = prompt_id.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid prompt ID format: {prompt_id}. "
                f"Expected format: category.name[.version]"
            )

        category_prefix = parts[0]
        name = parts[1]

        folder = self.CATEGORY_FOLDERS.get(category_prefix, category_prefix)
        return self.prompts_dir / folder / f"{name}.yaml"

    def _path_to_id(self, path: Path, folder: str) -> str:
        """Convert file path to prompt ID.

        Args:
            path: Path to the prompt file.
            folder: Folder name containing the file.

        Returns:
            Prompt identifier.
        """
        # Get category prefix from folder
        for prefix, folder_name in self.CATEGORY_FOLDERS.items():
            if folder_name == folder:
                name = path.stem
                return f"{prefix}.{name}"

        # Fall back to folder name as prefix
        return f"{folder}.{path.stem}"


def load_prompt(
    prompt_id: str, prompts_dir: Path | str = Path("prompts")
) -> PromptDefinition:
    """Convenience function to load a single prompt.

    Args:
        prompt_id: Prompt identifier.
        prompts_dir: Directory containing prompts.

    Returns:
        Loaded prompt definition.
    """
    loader = PromptLoader(prompts_dir)
    return loader.load(prompt_id)
