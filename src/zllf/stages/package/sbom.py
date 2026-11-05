"""AI-SBOM: a CycloneDX ML-BOM for one run (FR-09, P2.7.2).

Components:
- ``machine-learning-model``: student/base, teacher, judge, NLI, embedder (one per distinct
  model id, roles in properties) and the trained adapter (``metadata.component``);
- ``data``: synthetic train/val/test splits and the bundled replay/anchor benchmarks;
- ``library``: every installed Python distribution (``importlib.metadata``).

Model hashes are resolved offline: first from ``models/manifest.json`` (recorded at
download time), else from the local Hugging Face cache, where LFS blobs are named by their
sha256 so nothing large is re-read. A model's hash is the tree hash of its files, computed
like :func:`zllf.core.hashing.sha256_dir`. Dependencies record lineage: adapter → base model
+ train data + replay; synthetic data → teacher/NLI/embedder.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import uuid
import warnings
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path
from typing import Any

from cyclonedx.contrib.license.factories import LicenseFactory
from cyclonedx.model import HashAlgorithm, HashType, Property
from cyclonedx.model.bom import Bom
from cyclonedx.model.component import Component, ComponentType
from cyclonedx.output.json import JsonV1Dot6
from cyclonedx.schema import SchemaVersion
from packageurl import PackageURL

import zllf
from zllf.core.config import RunConfig
from zllf.core.hashing import sha256_dir, sha256_file
from zllf.core.manifest import RunPaths
from zllf.stages.package import artefacts as A
from zllf.stages.package.report import model_roles

SPEC_VERSION = SchemaVersion.V1_6
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LICENSES = LicenseFactory()


# ---------------------------------------------------------------- hashing helpers
def tree_hash(entries: Iterable[tuple[str, str]]) -> str:
    """sha256 over sorted ``(relative_path, sha256)`` pairs — same algorithm as sha256_dir."""
    h = hashlib.sha256()
    for rel, digest in sorted(entries):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(digest.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _hf_snapshot_dir(repo_id: str) -> Path | None:
    """Local snapshot dir of a cached HF repo (offline; ``None`` if not cached)."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    for probe in ("config.json", "README.md", "tokenizer_config.json", "modules.json"):
        try:
            found = try_to_load_from_cache(repo_id, probe)
        except Exception:  # invalid repo ids etc.
            return None
        if isinstance(found, str):
            return Path(found).parent
    return None


def hf_cache_files(snapshot: Path) -> list[tuple[str, str]]:
    """(relative path, sha256) for every file in a snapshot dir, cheaply where possible."""
    out = []
    for path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        blob = path.resolve().name if path.is_symlink() else ""
        digest = blob if _HEX64.match(blob) else sha256_file(path)
        out.append((path.relative_to(snapshot).as_posix(), digest))
    return out


def resolve_model(name: str, manifest_entry: dict[str, Any] | None) -> dict[str, Any]:
    """Hash/revision/licence info for a model id; ``resolved`` False if nothing is known."""
    if manifest_entry and isinstance(manifest_entry.get("files"), list):
        files = [(str(f["path"]), str(f["sha256"])) for f in manifest_entry["files"]
                 if isinstance(f, dict) and _HEX64.match(str(f.get("sha256", "")))]
        if files:
            return {"resolved": True, "source": "models/manifest.json",
                    "sha256": tree_hash(files), "n_files": len(files),
                    "revision": manifest_entry.get("revision"),
                    "license": manifest_entry.get("license"),
                    "size_bytes": manifest_entry.get("total_size")}
    if not name.startswith("fake") and (snapshot := _hf_snapshot_dir(name)) is not None:
        files = hf_cache_files(snapshot)
        return {"resolved": True, "source": "hf-cache", "sha256": tree_hash(files),
                "n_files": len(files), "revision": snapshot.name,
                "license": (manifest_entry or {}).get("license"), "size_bytes": None}
    return {"resolved": False, "source": "fake" if name.startswith("fake") else "unresolved",
            "revision": None, "license": (manifest_entry or {}).get("license")}


# ---------------------------------------------------------------- component builders
def _sha(digest: str) -> HashType:
    return HashType(alg=HashAlgorithm.SHA_256, content=digest)


def _props(**values: Any) -> list[Property]:
    return [Property(name=f"zllf:{k}", value=str(v)) for k, v in values.items() if v is not None]


def _license(value: str | None) -> list[Any]:
    if not value:
        return []
    try:
        return [_LICENSES.make_from_string(value)]
    except Exception:
        return []


def _hf_purl(name: str, revision: str | None) -> PackageURL | None:
    if "/" not in name or name.startswith("fake"):
        return None
    namespace, _, model = name.partition("/")
    return PackageURL(type="huggingface", namespace=namespace, name=model,
                      version=revision or None)


def model_component(name: str, roles: list[str], info: dict[str, Any]) -> Component:
    return Component(
        type=ComponentType.MACHINE_LEARNING_MODEL,
        name=name,
        version=info.get("revision") or None,
        bom_ref=f"model:{name}",
        hashes=[_sha(info["sha256"])] if info.get("sha256") else [],
        licenses=_license(info.get("license")),
        purl=_hf_purl(name, info.get("revision")),
        properties=_props(roles=",".join(sorted(roles)), hash_source=info["source"],
                          n_files=info.get("n_files"), size_bytes=info.get("size_bytes"),
                          hash_algorithm="tree-sha256(path,sha256)"),
    )


def data_component(ref: str, name: str, digest: str, description: str,
                   **props: Any) -> Component:
    return Component(type=ComponentType.DATA, name=name, bom_ref=f"data:{ref}",
                     hashes=[_sha(digest)], description=description, properties=_props(**props))


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def package_components() -> list[Component]:
    """One ``library`` component per installed Python distribution (deduplicated, sorted)."""
    seen: dict[str, Component] = {}
    for dist in metadata.distributions():
        raw_name = dist.metadata["Name"] if dist.metadata else None
        if not raw_name:
            continue
        name, version = _normalize(raw_name), dist.version
        if name in seen:
            continue
        lic = dist.metadata.get("License-Expression") or dist.metadata.get("License")
        lic = lic if lic and len(lic) <= 64 and "\n" not in lic else None
        seen[name] = Component(
            type=ComponentType.LIBRARY, name=name, version=version,
            bom_ref=f"pkg:pypi/{name}@{version}",
            purl=PackageURL(type="pypi", name=name, version=version),
            licenses=_license(lic),
        )
    return [seen[k] for k in sorted(seen)]


# ---------------------------------------------------------------- BOM
def build_sbom(
    paths: RunPaths,
    config: RunConfig,
    report: dict[str, Any],
    *,
    models_manifest: dict[str, dict[str, Any]] | None = None,
    bench: dict[str, Path] | None = None,
    include_packages: bool = True,
) -> Bom:
    """Assemble the ML-BOM from the run artefacts and the already-built audit report."""
    run_id = str(report["run_id"])
    adapter_sha = str(report["adapter"]["sha256"])
    created = report.get("created_at")
    bom = Bom(serial_number=uuid.uuid5(uuid.NAMESPACE_URL, f"zllf:{run_id}:{adapter_sha}"))
    if created:
        bom.metadata.timestamp = dt.datetime.fromisoformat(created)
    bom.metadata.tools.components.add(Component(
        type=ComponentType.APPLICATION, name="zllf", version=zllf.__version__,
        bom_ref="tool:zllf"))

    # models (one component per distinct model id)
    roles_by_model: dict[str, list[str]] = {}
    for role, name in model_roles(config).items():
        roles_by_model.setdefault(name, []).append(role)
    models_manifest = models_manifest or {}
    model_refs: dict[str, Component] = {}
    for name, roles in sorted(roles_by_model.items()):
        info = resolve_model(name, models_manifest.get(name))
        model_refs[name] = model_component(name, roles, info)
        bom.components.add(model_refs[name])

    # datasets
    data_refs: dict[str, Component] = {}
    synth = paths.stage("synth")
    for fname in A.SPLIT_FILES:
        path = synth / fname
        if path.exists():
            split = fname.removesuffix(".jsonl")
            comp = data_component(f"synth/{fname}", f"synthetic-{split}", sha256_file(path),
                                  f"Synthetic {split} split generated from masked documents",
                                  rows=A.jsonl_count(path), path=f"03_synth/{fname}")
            data_refs[split] = comp
            bom.components.add(comp)
    for name, directory in sorted((bench or {}).items()):
        comp = data_component(f"bench/{name}", f"bench-{name}", sha256_dir(directory),
                              f"Bundled {name} benchmark data", n_files=len(
                                  A.dir_files(directory)))
        data_refs[name] = comp
        bom.components.add(comp)

    # the adapter is the subject of this BOM
    m = config.models
    gates = report["evaluation"]["gates"]
    adapter = Component(
        type=ComponentType.MACHINE_LEARNING_MODEL,
        name=f"zllf-adapter-{run_id}", version=run_id, bom_ref="adapter",
        hashes=[_sha(adapter_sha)],
        description="LoRA adapter trained by ZLLF (hash: tree-sha256 of adapter/)",
        properties=_props(base_model=m.student, mode=config.mode,
                          size_bytes=report["adapter"]["size_bytes"],
                          ready=str(report["ready"]).lower(),
                          gates=",".join(f"{g['gate']}={'pass' if g['passed'] else 'fail'}"
                                         for g in gates),
                          seed=report["seeds"]["seed"],
                          config_sha256=report["config_sha256"]),
    )
    bom.metadata.component = adapter

    if include_packages:
        for comp in package_components():
            bom.components.add(comp)

    # lineage
    adapter_deps = [model_refs[m.student]]
    adapter_deps += [data_refs[k] for k in ("train", "val", "replay") if k in data_refs]
    bom.register_dependency(adapter, adapter_deps)
    generators = [model_refs[n] for n in (m.teacher, m.nli, m.embedder, m.judge or m.teacher)]
    generators = list({c.bom_ref.value: c for c in generators}.values())
    for split in ("train", "val", "test"):
        if split in data_refs:
            bom.register_dependency(data_refs[split], generators)
    return bom


def sbom_json(bom: Bom) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "incomplete dependency graph" notices
        return JsonV1Dot6(bom).output_as_string(indent=2)


def validate_sbom_json(text: str) -> str | None:
    """Validate against the bundled CycloneDX 1.6 JSON schema.

    Returns ``None`` if valid, an error message if invalid, and raises ``RuntimeError``
    if the validator's optional dependencies are unavailable.
    """
    from cyclonedx.validation.json import JsonStrictValidator

    try:
        error = JsonStrictValidator(SPEC_VERSION).validate_str(text)
    except Exception as exc:  # MissingOptionalDependencyException
        if "Missing" in type(exc).__name__ or isinstance(exc, ImportError):
            raise RuntimeError(f"CycloneDX JSON validation unavailable: {exc}") from exc
        raise
    return None if error is None else str(error)[:500]


def write_sbom(bom: Bom, path: Path) -> dict[str, Any]:
    """Serialize, validate (when possible) and write. Returns stats for the stage metrics."""
    text = sbom_json(bom)
    try:
        error = validate_sbom_json(text)
        validated = error is None
    except RuntimeError:
        error, validated = None, False
    if error is not None:
        from zllf.core.errors import ErrorCode, ZLLFError

        raise ZLLFError(ErrorCode.PACKAGE_FAILED, f"SBOM invalid: {error}")
    path.write_text(text + "\n", encoding="utf-8")
    data = json.loads(text)
    types: dict[str, int] = {}
    for comp in data.get("components", []):
        types[comp["type"]] = types.get(comp["type"], 0) + 1
    return {"spec_version": data.get("specVersion"), "validated": validated,
            "components_by_type": dict(sorted(types.items()))}
