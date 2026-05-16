"""MkDocs hooks for SAX-specific preprocessing."""

import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Any

import matplotlib.pyplot as plt

import sax

plt.rcParams.update(
    {
        "figure.figsize": (6, 2.5),
        "axes.grid": True,
        "lines.color": "grey",
        "patch.edgecolor": "grey",
        "text.color": "grey",
        "axes.facecolor": "ffffff00",
        "axes.edgecolor": "grey",
        "axes.labelcolor": "grey",
        "xtick.color": "grey",
        "ytick.color": "grey",
        "grid.color": "grey",
        "figure.facecolor": "ffffff00",
        "figure.edgecolor": "ffffff00",
        "savefig.facecolor": "ffffff00",
        "savefig.edgecolor": "ffffff00",
    }
)


def on_startup(command: str, dirty: bool, **kwargs: Any) -> None:
    """Called once when the MkDocs build starts."""


def on_shutdown(**kwargs: Any) -> None:
    """Called once when the MkDocs build ends."""


def on_serve(server: Any, config: Any, builder: Any, **kwargs: Any) -> None:
    """Called when the MkDocs development server starts."""


def on_config(config: Any, **kwargs: Any) -> Any:
    """Called after config file is loaded but before validation."""
    return config


def on_pre_build(config: Any, **kwargs: Any) -> None:
    """Called before the build starts."""


def on_files(files: Any, config: Any, **kwargs: Any) -> Any:
    """Called after files are gathered but before processing."""
    return files


def on_nav(nav: Any, config: Any, files: Any, **kwargs: Any) -> Any:
    """Called after navigation is built."""
    return nav


def on_env(env: Any, config: Any, files: Any, **kwargs: Any) -> Any:
    """Called after Jinja2 environment is created."""
    return env


def on_post_build(config: Any, **kwargs: Any) -> None:
    """Called after the build is complete."""


def on_pre_template(
    template: Any, template_name: str, config: Any, **kwargs: Any
) -> Any:
    """Called before a template is rendered."""
    return template


def on_template_context(
    context: Any, template_name: str, config: Any, **kwargs: Any
) -> Any:
    """Called after template context is created."""
    return context


def on_post_template(
    output: str, template_name: str, config: Any, **kwargs: Any
) -> str:
    """Called after template is rendered."""
    return output


def on_pre_page(page: Any, config: Any, files: Any, **kwargs: Any) -> Any:
    """Called before a page is processed."""
    return page


def on_page_read_source(page: Any, config: Any, **kwargs: Any) -> str | None:
    """Called to read the raw source of a page."""
    return None


def on_page_markdown(
    markdown: str, page: Any, config: Any, files: Any, **kwargs: Any
) -> str:
    """Process markdown content before it's converted to HTML."""
    blocks = markdown.split("```")

    for i, block in enumerate(blocks):
        if i % 2:
            # This is a code block
            if (special := _parse_special(block)) is not None:
                blocks[i] = special
            else:
                blocks[i] = f"```{block}```"
            continue

        # This is regular markdown content
        lines = block.split("\n")
        _insert_cross_refs(lines)
        blocks[i] = "\n".join(lines)

    content = "".join(blocks)
    return content


def on_page_content(  # noqa: C901
    html: str, page: Any, config: Any, files: Any, **kwargs: Any
) -> str:
    """Called after markdown is converted to HTML."""
    if "```{svgbob}" not in html:
        return html

    source_parts = []
    for part in html.split("```"):
        if not part.startswith("{svgbob}"):
            continue
        lines = part.split("\n")[1:]
        for i, line in enumerate(lines):
            lines[i] = "".join(re.split(r"[<>]", line)[::2])
        part = dedent("\n".join(lines))
        source_parts.append(lines)

    rendered_parts = []
    for i, part in enumerate(html.split('<div class="language-text highlight">')):
        if i > 0:
            part = f'<div class="language-text highlight">{part}'
        first, *rest = part.split("</span></code></pre></div>")
        rest = "</span></code></pre></div>".join(rest)
        first = f"{first}</span></code></pre></div>"
        rendered_parts.append(first)
        rendered_parts.append(rest)

    for i, part in enumerate(rendered_parts):
        if not part.startswith('<div class="language-text highlight">'):
            continue
        lines = part.split("\n")
        if (svgbob_source := _svgbob_source(lines, source_parts)) is None:
            continue
        if (svg := _svgbob_svg(svgbob_source)) is None:
            continue
        rendered_parts[i] = svg

    return "".join(rendered_parts)


def on_page_context(
    context: Any, page: Any, config: Any, nav: Any, **kwargs: Any
) -> Any:
    """Called after page context is created."""
    return context


def on_post_page(output: str, page: Any, config: Any, **kwargs: Any) -> str:
    """Called after page is fully processed."""
    return output


def _parse_special(content: str) -> str | None:
    """Format contents of a special code block differently."""
    lines = content.strip().split("\n")
    first = lines[0].strip()
    rest = lines[1:]
    if not (first.startswith("{") and first.endswith("}")):
        return None
    code_block_type = first[1:-1].strip()
    if code_block_type == "svgbob":
        source = "\n".join(rest)
        content_hash = hashlib.md5(source.encode()).hexdigest()
        svg_content = _svgbob_svg(source)
        if not svg_content:
            return None
        docs_dir = Path(__file__).parent
        svg_path = docs_dir / "assets" / "svgbob" / f"svgbob_{content_hash}.svg"
        svg_path.parent.mkdir(exist_ok=True, parents=True)
        svg_path.write_text(svg_content or "")
        return f"\n\n![{svg_path.name}](/sax/{svg_path.relative_to(docs_dir)})\n\n"
    return _format_admonition(code_block_type, rest)


def _format_admonition(admonition_type: str, lines: list[str]) -> str:
    """Format lines as an admonition."""
    if admonition_type == "hint":
        admonition_type = "info"
    ret = f"!!! {admonition_type}\n\n"
    for line in lines:
        ret += f"    {line.strip()}\n"
    return ret


def _svgbob_svg(source: str) -> str | None:
    source = source.replace("&lt;", "<").replace("&gt;", ">")
    svgbob = shutil.which("svgbob_cli")
    if not svgbob:
        print("Warning: svgbob_cli is not installed or not found in PATH.")  # noqa: T201
        return None
    content_hash = hashlib.md5(source.encode()).hexdigest()
    txt_filename = f"svgbob_{content_hash}.txt"
    temp_path = Path(tempfile.gettempdir()).resolve() / "svgbob" / txt_filename
    temp_path.parent.mkdir(exist_ok=True)
    try:
        temp_path.write_text(source)
        return subprocess.check_output(  # noqa: S603
            [
                svgbob,
                "--background",
                "#00000000",
                "--stroke-color",
                "grey",
                "--fill-color",
                "grey",
                str(temp_path),
            ]
        ).decode()
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Error generating SVG with svgbob: {e}")  # noqa: T201
        return None
    finally:
        temp_path.unlink()


def _svgbob_source(
    rendered_lines: list[str], source_parts: list[list[str]]
) -> str | None:
    for source_lines in source_parts:
        if all(
            sl.strip() in rl
            for sl, rl in zip(source_lines, rendered_lines, strict=False)
        ):
            return "\n".join(source_lines)
    return None


def _insert_cross_refs(lines: list[str]) -> None:  # noqa: C901
    """Insert cross-references in the markdown lines."""
    cross_refs = {}
    for line in lines:
        parts = line.split("`")
        for k, part in enumerate(parts):
            if k % 2 == 0:
                continue
            *first, short_part = part.split(".")
            if first and first[0] != "sax":
                continue
            if hasattr(sax, short_part):
                cross_refs[f"`{part}`"] = f"[`{part}`][sax.{short_part}]"
            elif hasattr(sax.fit, short_part):
                cross_refs[f"`{part}`"] = f"[`{part}`][sax.fit.{short_part}]"
            elif hasattr(sax.models, part):
                cross_refs[f"`{part}`"] = f"[`{part}`][sax.models.{short_part}]"
            elif hasattr(sax.parsers, part):
                cross_refs[f"`{part}`"] = f"[`{part}`][sax.parsers.{short_part}]"
    for j, line in enumerate(lines):
        for k, v in cross_refs.items():
            line = line.replace(k, v)
        lines[j] = line
