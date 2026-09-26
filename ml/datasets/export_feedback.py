"""Export the app's own collected data so training can pick it up.

Two outputs:

* ``<DATA_ROOT>/own_photos/<Crop___Condition>/<task id>.<ext>`` - every :class:`LabelTask` that two
  labellers agreed on **and** that carries consent. Files are copied (never moved) and the task is
  marked ``exported`` so a second run does not duplicate work. ``own_photos`` is the ``local_dir``
  source registered in ``ml/datasets/sources.yaml``, so the manifest builder picks it up as-is.
* ``data/llm/feedback.jsonl`` - one line per rated chat turn
  ``{session_key, user, assistant, rating, comment, language}`` for assistant evaluation / tuning.

Usage::

    python -m ml.datasets.export_feedback --out <DATA_ROOT>/own_photos
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from flask import current_app

from . import data_root

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


# --------------------------------------------------------------------------- images
def export_images(out_dir: str | Path) -> dict:
    """Copy every agreed, consented photo into ``out_dir/<final_label>/``."""
    from krishidisha.blueprints.feedback import resolve_image
    from krishidisha.extensions import db
    from krishidisha.models import LabelTask

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = LabelTask.query.filter(LabelTask.status == "labelled", LabelTask.consent.is_(True)) \
        .order_by(LabelTask.id).all()

    exported, skipped, missing = 0, 0, 0
    per_class: dict[str, int] = {}
    for task in tasks:
        label = (task.final_label or "").strip()
        if not label or "___" not in label:
            skipped += 1
            continue
        src = resolve_image(task.image_path)
        if src is None or not src.is_file():
            missing += 1
            continue
        ext = src.suffix.lower()
        dest_dir = out / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest_dir / f"{task.id}{ext if ext in IMAGE_EXT else '.jpg'}")
        task.status = "exported"
        per_class[label] = per_class.get(label, 0) + 1
        exported += 1
    db.session.commit()
    return {"exported": exported, "skipped": skipped, "missing": missing, "per_class": per_class,
            "out_dir": str(out), "classes": len(per_class)}


# --------------------------------------------------------------------------- chat turns
def _turn_for(chat, ref_id: str):
    """(user_message, assistant_message) for ``<session_key>#<index>``; falls back to the last turn."""
    messages = list(chat.messages)
    if not messages:
        return None, None
    _, _, idx = (ref_id or "").partition("#")
    position = None
    if idx.isdigit() and 0 <= int(idx) < len(messages) and messages[int(idx)].role == "assistant":
        position = int(idx)
    else:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "assistant":
                position = i
                break
    if position is None:
        return None, None
    user = messages[position - 1] if position > 0 and messages[position - 1].role == "user" else None
    return user, messages[position]


def export_chat_feedback(out_file: str | Path) -> dict:
    """Write every consented chat rating as JSONL."""
    from krishidisha.models import ChatSession, Feedback

    path = Path(out_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = Feedback.query.filter(Feedback.kind == "chat", Feedback.consent.is_(True)) \
        .order_by(Feedback.id).all()

    written = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            key = (row.ref_id or "").partition("#")[0]
            chat = ChatSession.query.filter_by(session_key=key).first()
            if chat is None:
                continue
            user, assistant = _turn_for(chat, row.ref_id)
            if assistant is None:
                continue
            fh.write(json.dumps({
                "session_key": key,
                "user": user.content if user else None,
                "assistant": assistant.content,
                "rating": row.rating,
                "comment": row.comment,
                "language": row.language or chat.language,
            }, ensure_ascii=False) + "\n")
            written += 1
    return {"chat_turns": written, "jsonl": str(path)}


# --------------------------------------------------------------------------- entry points
def run_export(out_dir: str | Path | None = None, jsonl: str | Path | None = None) -> dict:
    """Run both exports inside an active Flask app context. Reads ``KRISHIDISHA_DATA_ROOT`` now."""
    out = Path(out_dir) if out_dir else data_root() / "own_photos"
    target = Path(jsonl) if jsonl else Path(current_app.config["DATA_DIR"]) / "llm" / "feedback.jsonl"
    summary = export_images(out)
    summary.update(export_chat_feedback(target))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="default: <DATA_ROOT>/own_photos")
    parser.add_argument("--jsonl", type=Path, default=None, help="default: data/llm/feedback.jsonl")
    args = parser.parse_args(argv)

    from krishidisha import create_app

    app = create_app()
    with app.app_context():
        summary = run_export(args.out, args.jsonl)
    print(f"exported {summary['exported']} photo(s) into {summary['out_dir']} "
          f"across {summary['classes']} class(es)")
    if summary["skipped"] or summary["missing"]:
        print(f"  skipped (no usable final label): {summary['skipped']}; missing files: {summary['missing']}")
    for label, n in sorted(summary["per_class"].items()):
        print(f"  {label}: {n}")
    print(f"wrote {summary['chat_turns']} rated chat turn(s) to {summary['jsonl']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
