"""
Manifest data structures and validation.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# Required columns that MUST be present in every manifest
REQUIRED_COLUMNS = ["sample_id", "media_id", "label", "file_path", "split"]

# Preferred/optional columns for richer metadata
PREFERRED_COLUMNS = [
    "dataset",
    "identity_id",
    "source_id",
    "manipulation_type",
    "generator",
    "sha256",
]

VALID_LABELS = {0, 1}  # 0 = AUTHENTIC, 1 = MANIPULATED
VALID_SPLITS = {"train", "val", "test"}


@dataclass
class Manifest:
    """
    Versioned dataset manifest containing sample metadata.

    The manifest preserves source-video and identity relationships
    to enable leakage-controlled splits.
    """
    dataframe: pd.DataFrame
    version: str = "1.0"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate manifest after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate manifest structure and content."""
        # Check required columns
        missing = set(REQUIRED_COLUMNS) - set(self.dataframe.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")

        # Check for empty dataframe
        if len(self.dataframe) == 0:
            raise ValueError("Manifest is empty")

        # Validate labels
        invalid_labels = set(self.dataframe["label"].unique()) - VALID_LABELS
        if invalid_labels:
            raise ValueError(f"Invalid labels found: {invalid_labels}. Valid labels: {VALID_LABELS}")

        # Validate splits
        invalid_splits = set(self.dataframe["split"].unique()) - VALID_SPLITS
        if invalid_splits:
            raise ValueError(f"Invalid splits found: {invalid_splits}. Valid splits: {VALID_SPLITS}")

        # Check for duplicate sample_ids
        if self.dataframe["sample_id"].duplicated().any():
            raise ValueError("Duplicate sample_ids found in manifest")

        # Check for duplicate file_paths
        if self.dataframe["file_path"].duplicated().any():
            raise ValueError("Duplicate file_paths found in manifest")

        # Validate no cross-split leakage by media_id
        self._validate_split_exclusivity("media_id", "media_id")

        # Validate no cross-split leakage by identity_id (if column exists)
        if "identity_id" in self.dataframe.columns:
            self._validate_split_exclusivity("identity_id", "identity_id")

        # Validate no cross-split leakage by sha256 (if column exists)
        if "sha256" in self.dataframe.columns:
            # Only validate non-dummy hashes
            valid_hashes = self.dataframe["sha256"].notna() & (self.dataframe["sha256"] != "") & ~self.dataframe["sha256"].str.startswith("dummy_hash")
            if valid_hashes.any():
                self._validate_split_exclusivity("sha256", "sha256", mask=valid_hashes)

        # Verify file paths exist (warning only, not error)
        missing_files = []
        for _, row in self.dataframe.iterrows():
            if not Path(row["file_path"]).exists():
                missing_files.append(row["file_path"])
        if missing_files:
            import warnings
            warnings.warn(f"{len(missing_files)} files not found in manifest (first 5: {missing_files[:5]})", stacklevel=2)

    def _validate_split_exclusivity(self, column: str, error_name: str, mask: pd.Series | None = None) -> None:
        """Validate that each unique value in column appears in exactly one split."""
        df = self.dataframe
        if mask is not None:
            df = df[mask]
        
        # Group by column value and count unique splits
        split_counts = df.groupby(column)["split"].nunique()
        violating = split_counts[split_counts > 1]
        
        if len(violating) > 0:
            # Build detailed error message
            details = []
            for val in violating.index:
                splits = df[df[column] == val]["split"].unique()
                details.append(f"  {val}: {list(splits)}")
            raise ValueError(
                f"{error_name} appears in multiple splits (leakage detected):\n" + "\n".join(details)
            )

    def get_split(self, split: str) -> "Manifest":
        """Return a new Manifest filtered to a specific split."""
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split: {split}. Valid: {VALID_SPLITS}")
        df = self.dataframe[self.dataframe["split"] == split].reset_index(drop=True)
        return Manifest(dataframe=df, version=self.version, description=self.description, metadata=self.metadata)

    def get_by_media_id(self, media_id: str) -> "Manifest":
        """Return samples belonging to a specific media_id."""
        df = self.dataframe[self.dataframe["media_id"] == media_id].reset_index(drop=True)
        return Manifest(dataframe=df, version=self.version, description=self.description, metadata=self.metadata)

    def get_by_identity(self, identity_id: str) -> "Manifest":
        """Return samples belonging to a specific identity_id."""
        if "identity_id" not in self.dataframe.columns:
            raise ValueError("Manifest does not contain identity_id column")
        df = self.dataframe[self.dataframe["identity_id"] == identity_id].reset_index(drop=True)
        return Manifest(dataframe=df, version=self.version, description=self.description, metadata=self.metadata)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> pd.Series:
        return self.dataframe.iloc[idx]

    def to_csv(self, path: str | Path) -> None:
        """Save manifest to CSV."""
        self.dataframe.to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs) -> "Manifest":
        """Load manifest from CSV file."""
        df = pd.read_csv(path)
        return cls(dataframe=df, **kwargs)

    def compute_hash(self) -> str:
        """Compute SHA256 hash of the manifest dataframe for versioning."""
        content = self.dataframe.to_csv(index=False).encode()
        return hashlib.sha256(content).hexdigest()[:16]


def load_manifest(path: str | Path, **kwargs) -> Manifest:
    """Load and validate a manifest from CSV file."""
    return Manifest.from_csv(path, **kwargs)


def create_manifest_template() -> pd.DataFrame:
    """Create an empty DataFrame with the correct columns for a manifest."""
    all_columns = REQUIRED_COLUMNS + PREFERRED_COLUMNS
    return pd.DataFrame(columns=all_columns)
