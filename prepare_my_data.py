"""Prepare GAR-Font base-model data into ./my_data (does not touch ./data).

Keeps only fonts that cover enough frequently-used Chinese characters, then
uses their intersection as the charset so every train/test style has every glyph.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from font2data.FontData import FontData
from font2data.words import special_dict
from tqdm import tqdm

FREQUENTLY_USED = special_dict["frequently_used"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fonts-dir", type=Path, default=Path("fonts"))
    p.add_argument("--out-dir", type=Path, default=Path("my_data"))
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--pad", type=int, default=4)
    p.add_argument("--train-ratio", type=float, default=0.8, help="char train split")
    p.add_argument("--test-font-ratio", type=float, default=0.1)
    p.add_argument("--n-ref", type=int, default=8)
    p.add_argument("--content-ref", type=str, default="1")
    p.add_argument("--min-freq-coverage", type=float, default=0.85,
                   help="keep fonts covering at least this fraction of frequently_used")
    p.add_argument("--min-charset", type=int, default=2000,
                   help="drop weakest fonts until intersection reaches this size")
    p.add_argument("--min-fonts", type=int, default=30)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--scan-only", action="store_true")
    return p.parse_args()


def discover_font_dirs(fonts_dir: Path) -> list[Path]:
    dirs = sorted(
        [p for p in fonts_dir.iterdir() if p.is_dir() and (p / "metadata.json").exists()],
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    )
    if not dirs:
        raise FileNotFoundError(f"No font folders with metadata.json under {fonts_dir}")
    return dirs


def scan_one_font(font_dir: str) -> tuple[str, list[int] | None, str | None]:
    name = Path(font_dir).name
    try:
        fd = FontData(font_dir, font_size=32, debug=False)
        freq = sorted(fd.get_supported_chars() & FREQUENTLY_USED)
        return name, freq, None
    except Exception as e:
        return name, None, f"{type(e).__name__}: {e}"


def load_or_scan_coverage(
    font_dirs: list[Path],
    cache_path: Path,
    workers: int,
) -> dict[str, set[int]]:
    cache: dict[str, list[int]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Loaded coverage cache: {cache_path} ({len(cache)} fonts)")

    pending = [d for d in font_dirs if d.name not in cache]
    failed: dict[str, str] = {}
    if pending:
        print(f"Scanning frequently_used coverage for {len(pending)} fonts...")
        with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(scan_one_font, str(d)): d.name for d in pending}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Scan fonts"):
                name, freq, err = fut.result()
                if err or freq is None:
                    failed[name] = err or "unknown"
                    print(f"  skip {name}: {failed[name]}")
                    continue
                cache[name] = freq
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
        print(f"Wrote coverage cache ({len(cache)} fonts, {len(failed)} failed)")

    return {name: set(cps) for name, cps in cache.items()}


def select_fonts_and_charset(
    coverage: dict[str, set[int]],
    content_ref: str,
    min_freq_coverage: float,
    min_charset: int,
    min_fonts: int,
) -> tuple[list[str], list[str]]:
    n_freq = len(FREQUENTLY_USED)
    threshold = int(n_freq * min_freq_coverage)

    ranked = sorted(coverage.items(), key=lambda kv: len(kv[1]), reverse=True)
    print(f"frequently_used={n_freq}, coverage threshold={threshold} ({min_freq_coverage:.0%})")
    print("Top 5 coverage:", [(n, len(s)) for n, s in ranked[:5]])
    print("Bottom 5 coverage:", [(n, len(s)) for n, s in ranked[-5:]])

    selected = {n: s for n, s in coverage.items() if len(s) >= threshold}
    if content_ref not in selected:
        if content_ref in coverage:
            selected[content_ref] = coverage[content_ref]
            print(f"Force-included content-ref {content_ref} "
                  f"(freq={len(coverage[content_ref])})")
        else:
            raise RuntimeError(f"content-ref {content_ref} has no coverage (scan failed?)")

    excluded = sorted(set(coverage) - set(selected), key=lambda n: int(n) if n.isdigit() else n)
    print(f"Fonts passing coverage: {len(selected)}; excluded: {len(excluded)}")

    def intersection_of(names: list[str]) -> set[int]:
        return set.intersection(*(selected[n] for n in names))

    names = list(selected)
    inter = intersection_of(names)
    print(f"Initial intersection={len(inter)}")

    # Drop fonts (never content-ref) that most enlarge the intersection.
    while len(inter) < min_charset and len(names) > min_fonts:
        best_drop = None
        best_size = len(inter)
        for n in names:
            if n == content_ref:
                continue
            trial = [x for x in names if x != n]
            sz = len(intersection_of(trial))
            if sz > best_size:
                best_size = sz
                best_drop = n
        if best_drop is None:
            break
        names.remove(best_drop)
        excluded.append(best_drop)
        inter = intersection_of(names)
        print(f"  dropped {best_drop} -> fonts={len(names)}, intersection={len(inter)}")

    if len(inter) < min_charset:
        print(f"Warning: intersection {len(inter)} < min_charset {min_charset}")

    chars = [chr(cp) for cp in sorted(inter)]
    font_names = sorted(names, key=lambda n: int(n) if n.isdigit() else n)
    print(f"Selected {len(font_names)} fonts, charset={len(chars)}")
    return font_names, chars


def split_contents(chars: list[str], train_ratio: float, n_ref: int) -> dict:
    if len(chars) < n_ref + 2:
        raise RuntimeError(
            f"charset only {len(chars)}, need at least n_ref+2={n_ref + 2}"
        )
    n_train = max(n_ref + 1, int(len(chars) * train_ratio))
    n_train = min(n_train, len(chars) - 1)
    all_items = [{"char": ch, "index": i} for i, ch in enumerate(chars)]
    return {
        "all_content": all_items,
        "train_content": all_items[:n_train],
        "test_content": all_items[n_train:],
    }


def split_styles(font_names: list[str], content_ref: str, test_font_ratio: float) -> dict:
    if content_ref not in font_names:
        raise ValueError(f"--content-ref {content_ref} not in selected fonts")
    others = [n for n in font_names if n != content_ref]
    n_test = max(1, int(len(others) * test_font_ratio)) if others else 0
    test_style = others[-n_test:] if n_test else []
    train_style = [content_ref] + others[:-n_test] if n_test else [content_ref] + others
    return {
        "basic_style": [content_ref],
        "train_style": train_style,
        "test_style": test_style,
        "content_ref_style": [content_ref],
    }


def render_one_font(
    font_dir: str,
    style_name: str,
    chars: list[str],
    out_dir: str,
    image_size: int,
    pad: int,
) -> tuple[str, list[tuple[int, str]]]:
    fd = FontData(font_dir, font_size=image_size, debug=False)
    style_dir = Path(out_dir) / style_name
    style_dir.mkdir(parents=True, exist_ok=True)
    missing: list[tuple[int, str]] = []
    for idx, ch in enumerate(chars):
        dest = style_dir / f"{idx:04d}.png"
        if dest.exists():
            continue
        img = fd.char2img(ch, pad=pad)
        if img is None:
            missing.append((idx, ch))
            continue
        img.save(dest)
    return style_name, missing


def render_glyphs(
    fonts_dir: Path,
    font_names: list[str],
    chars: list[str],
    out_img_dir: Path,
    image_size: int,
    pad: int,
    workers: int,
) -> None:
    out_img_dir.mkdir(parents=True, exist_ok=True)
    n_need = len(chars)
    todo = []
    for name in font_names:
        style_dir = out_img_dir / name
        have = len(list(style_dir.glob("*.png"))) if style_dir.exists() else 0
        if have >= n_need:
            continue
        todo.append(name)
    print(f"Render {len(todo)} fonts ({len(font_names) - len(todo)} already complete), "
          f"{n_need} glyphs each, workers={workers}")
    if not todo:
        return

    failed = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {
            ex.submit(
                render_one_font,
                str(fonts_dir / name),
                name,
                chars,
                str(out_img_dir),
                image_size,
                pad,
            ): name
            for name in todo
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Render fonts"):
            name, missing = fut.result()
            if missing:
                failed.append((name, missing[:5], len(missing)))
                print(f"  {name} missing {len(missing)} e.g. {missing[:5]}")
    if failed:
        raise RuntimeError(f"{len(failed)} fonts failed to render all glyphs: {failed[:10]}")


def main() -> None:
    args = parse_args()
    if args.out_dir.resolve() == Path("data").resolve():
        raise SystemExit("Refusing to write into ./data")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    font_dirs = discover_font_dirs(args.fonts_dir)
    print(f"Discovered {len(font_dirs)} font folders")

    coverage = load_or_scan_coverage(
        font_dirs, args.out_dir / "_coverage_cache.json", args.workers
    )
    font_names, chars = select_fonts_and_charset(
        coverage,
        args.content_ref,
        args.min_freq_coverage,
        args.min_charset,
        args.min_fonts,
    )
    content_info = split_contents(chars, args.train_ratio, args.n_ref)
    style_info = split_styles(font_names, args.content_ref, args.test_font_ratio)

    report = {
        "n_discovered": len(font_dirs),
        "n_scanned": len(coverage),
        "n_selected": len(font_names),
        "charset": len(chars),
        "train_chars": len(content_info["train_content"]),
        "test_chars": len(content_info["test_content"]),
        "train_styles": len(style_info["train_style"]),
        "test_styles": len(style_info["test_style"]),
        "content_ref": args.content_ref,
        "min_freq_coverage": args.min_freq_coverage,
        "selected_fonts": font_names,
    }
    (args.out_dir / "prepare_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "split_content_info.json").write_text(
        json.dumps(content_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "split_style_info.json").write_text(
        json.dumps(style_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "selected_fonts"}, indent=2))

    if args.scan_only:
        print("scan-only: skip rendering")
        return

    render_glyphs(
        args.fonts_dir,
        font_names,
        chars,
        args.out_dir / "fontimg",
        args.image_size,
        args.pad,
        args.workers,
    )
    print("Wrote:")
    print(f"  {args.out_dir / 'fontimg'}/<id>/####.png")
    print(f"  {args.out_dir / 'split_content_info.json'}")
    print(f"  {args.out_dir / 'split_style_info.json'}")
    print("\nTrain with:")
    print(f"  --data-dir-path {args.out_dir / 'fontimg'}")
    print(f"  --data-style-info-json {args.out_dir / 'split_style_info.json'}")
    print(f"  --data-content-info-json {args.out_dir / 'split_content_info.json'}")


if __name__ == "__main__":
    main()
