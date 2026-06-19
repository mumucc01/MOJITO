#!/usr/bin/env python3
"""Remove Chinese text from comments and docstrings in project source files."""
import re
import sys
import tokenize
from io import BytesIO
from pathlib import Path

CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")

ROOTS = [
    Path(__file__).resolve().parents[1] / "MOJITO",
    Path(__file__).resolve().parents[1] / "Diffusion-Planner",
    Path(__file__).resolve().parents[1] / "setup_env.sh",
    Path(__file__).resolve().parents[1] / "scripts",
]

SKIP_PARTS = {
    "dinov3/dinov3",
    "Uni3D/Pointnet2",
    "__MACOSX",
    ".ipynb_checkpoints",
    "nuplan-devkit",
    "node_modules",
    ".git",
}


def has_cjk(text: str) -> bool:
    return CJK.search(text) is not None


def strip_cjk(text: str) -> str:
    return CJK.sub("", text)


def strip_cjk_lines(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and has_cjk(line):
            cleaned.append("")
            continue
        if "#" in line and has_cjk(line):
            code, _, comment = line.partition("#")
            if not has_cjk(code):
                cleaned.append(code.rstrip())
                continue
        if has_cjk(line):
            cleaned.append(strip_cjk(line))
        else:
            cleaned.append(line)
    return "\n".join(lines)


def is_docstring_token(tok: tokenize.TokenInfo, prev: tokenize.TokenInfo | None) -> bool:
    if tok.type != tokenize.STRING:
        return False
    if prev is None:
        return True
    if prev.type in (tokenize.NEWLINE, tokenize.INDENT):
        return True
    if prev.type == tokenize.NAME and prev.string in ("def", "class"):
        return True
    return False


def clean_string_token(s: str) -> str:
    quote = s[:3] if s[:3] in ("'''", '"""') else s[:1]
    end_quote = quote
    inner = s[len(quote):-len(end_quote)]
    inner = strip_cjk_lines(inner)
    inner = re.sub(r"\n{3,}", "\n\n", inner)
    return f"{quote}{inner}{end_quote}"


def process_python(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    try:
        tokens = list(tokenize.tokenize(BytesIO(source.encode("utf-8")).readline))
    except tokenize.TokenError:
        return False

    lines = source.splitlines(keepends=True)
    if not lines and source:
        lines = [source]

    changed = False
    prev: tokenize.TokenInfo | None = None
    for tok in tokens:
        if tok.type == tokenize.COMMENT and has_cjk(tok.string):
            row, col = tok.start
            idx = row - 1
            line = lines[idx]
            end_col = tok.end[1]
            prefix = line[:col]
            suffix = line[end_col:]
            if prefix.strip() == "" or prefix.strip() == "#":
                new_line = suffix if suffix.strip() else ""
            else:
                new_line = prefix.rstrip()
                if line.endswith("\n") and not new_line.endswith("\n"):
                    new_line += "\n"
            if new_line != line:
                lines[idx] = new_line
                changed = True
        elif tok.type == tokenize.STRING and has_cjk(tok.string):
            multiline = tok.start[0] != tok.end[0]
            docstring = is_docstring_token(tok, prev)
            if multiline or docstring:
                new_s = clean_string_token(tok.string)
                if new_s != tok.string:
                    row, col = tok.start
                    idx = row - 1
                    line = lines[idx]
                    if multiline:
                        end_row = tok.end[0] - 1
                        new_inner_lines = new_s.splitlines(keepends=True)
                        if tok.string.startswith("'''") or tok.string.startswith('"""'):
                            # Replace span of lines with cleaned multiline string token text
                            first = line[:col] + new_inner_lines[0]
                            lines[idx] = first
                            for mid in range(idx + 1, end_row):
                                lines[mid] = ""
                            if end_row > idx:
                                last = new_inner_lines[-1] + lines[end_row][tok.end[1]:]
                                lines[end_row] = last
                                for j, mid_line in enumerate(new_inner_lines[1:-1], start=idx + 1):
                                    lines[j] = mid_line
                            changed = True
                    else:
                        lines[idx] = line[:col] + new_s + line[tok.end[1]:]
                        changed = True
        if tok.type not in (tokenize.NL, tokenize.COMMENT, tokenize.ENCODING):
            prev = tok

    new_source = "".join(lines)
    if new_source != source:
        path.write_text(new_source, encoding="utf-8")
        return True
    return False


def process_text_comment_file(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    changed = False
    new_lines = []
    for line in lines:
        if "#" in line and has_cjk(line):
            hash_idx = line.find("#")
            code = line[:hash_idx]
            if code.strip() == "" or not has_cjk(code):
                if code.strip() == "":
                    new_line = "\n" if line.endswith("\n") else ""
                else:
                    new_line = code.rstrip() + ("\n" if line.endswith("\n") else "")
                if new_line != line:
                    changed = True
                new_lines.append(new_line)
                continue
        new_lines.append(line)
    if changed:
        path.write_text("".join(new_lines), encoding="utf-8")
    return changed


def main():
    paths = []
    for root in ROOTS:
        if root.is_file():
            paths.append(root)
        else:
            for ext in ("*.py", "*.sh", "*.yml", "*.yaml"):
                paths.extend(root.rglob(ext))

    updated = []
    for path in sorted(set(paths)):
        if any(part in str(path) for part in SKIP_PARTS):
            continue
        try:
            if path.suffix == ".py":
                if process_python(path):
                    updated.append(path)
            elif path.suffix in (".sh", ".yml", ".yaml"):
                if process_text_comment_file(path):
                    updated.append(path)
        except Exception as e:
            print(f"SKIP {path}: {e}", file=sys.stderr)

    for p in updated:
        print(p)
    print(f"Updated {len(updated)} files")


if __name__ == "__main__":
    main()
