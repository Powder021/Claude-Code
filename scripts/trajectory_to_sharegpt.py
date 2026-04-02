#!/usr/bin/env python3
"""
Convert Claude Code JSONL transcripts to ShareGPT training format.

Usage:
    python3 trajectory_to_sharegpt.py                           # all sessions in current project
    python3 trajectory_to_sharegpt.py --session <uuid>          # specific session
    python3 trajectory_to_sharegpt.py --input /path/to/file.jsonl  # specific file
    python3 trajectory_to_sharegpt.py -o output.json            # custom output path
    python3 trajectory_to_sharegpt.py --project-dir /path/to/dir   # custom project dir
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def parse_jsonl(path: str) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def build_conversation_chain(entries: list[dict], include_sidechains: bool = False) -> list[dict]:
    """Build ordered conversation from JSONL entries.

    The JSONL is written in chronological order, but the uuid/parentUuid
    structure forms a tree (parallel tool calls branch out). We use insertion
    order (which is chronological) and filter out non-message entries.

    For main sessions, sidechains are excluded. For subagent files,
    all entries are sidechains so include_sidechains must be True.
    """
    message_types = {"user", "assistant", "attachment"}
    messages = [
        e for e in entries
        if e.get("type") in message_types
        and e.get("message")
        and (include_sidechains or not e.get("isSidechain"))
    ]
    return messages


def format_content_blocks(blocks: list[dict]) -> tuple[str, str]:
    """Format content blocks into text and determine role.

    Returns (role, formatted_text) where role is 'gpt' or 'tool'.
    """
    parts = []
    role = "gpt"

    for block in blocks:
        btype = block.get("type")

        if btype == "text":
            text = block.get("text", "")
            if text.strip():
                parts.append(text)

        elif btype == "thinking":
            thinking = block.get("thinking", "")
            if thinking.strip():
                parts.append(f"<thinking>\n{thinking}\n</thinking>")

        elif btype == "redacted_thinking":
            # Skip redacted thinking
            continue

        elif btype == "tool_use":
            name = block.get("name", "unknown")
            arguments = block.get("input", {})
            tool_call = json.dumps(
                {"name": name, "arguments": arguments},
                ensure_ascii=False,
                indent=2,
            )
            parts.append(f"<tool_call>\n{tool_call}\n</tool_call>")

        elif btype == "tool_result":
            role = "tool"
            content = block.get("content", "")
            tool_use_id = block.get("tool_use_id", "")
            is_error = block.get("is_error", False)

            # content can be string or list of blocks
            if isinstance(content, list):
                text_parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        text_parts.append(c.get("text", ""))
                    elif isinstance(c, str):
                        text_parts.append(c)
                content_str = "\n".join(text_parts)
            else:
                content_str = str(content)

            tag = "tool_error" if is_error else "tool_response"
            parts.append(f"<{tag}>\n{content_str}\n</{tag}>")

    return role, "\n\n".join(parts)


def format_user_content(content: Any) -> str:
    """Format user message content (can be string or blocks)."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        role, text = format_content_blocks(content)
        return text

    return str(content)


def merge_consecutive_turns(turns: list[dict]) -> list[dict]:
    """Merge consecutive messages from the same role."""
    if not turns:
        return []

    merged = [turns[0].copy()]
    for turn in turns[1:]:
        if turn["from"] == merged[-1]["from"]:
            merged[-1]["value"] = merged[-1]["value"] + "\n\n" + turn["value"]
        else:
            merged.append(turn.copy())

    return merged


def try_parse_stringified_content(content: str) -> list[dict] | None:
    """Try to parse content that was stringified as Python repr or JSON."""
    if not isinstance(content, str) or not content.startswith("["):
        return None
    # Try JSON first
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    # Try Python literal (e.g. [{'type': 'text', ...}])
    import ast
    try:
        parsed = ast.literal_eval(content)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return None


def convert_session(entries: list[dict], session_id: str, is_subagent: bool = False) -> dict | None:
    """Convert a single session's entries to ShareGPT format."""
    chain = build_conversation_chain(entries, include_sidechains=is_subagent)
    if not chain:
        return None

    turns = []
    for entry in chain:
        msg = entry.get("message", {})
        msg_role = msg.get("role", "")
        content = msg.get("content", "")
        entry_type = entry.get("type", "")

        # Handle stringified content (common in subagent transcripts)
        if isinstance(content, str):
            parsed = try_parse_stringified_content(content)
            if parsed is not None:
                content = parsed

        if entry_type == "user":
            if isinstance(content, list):
                # Could be tool_result blocks
                role, text = format_content_blocks(content)
                if text.strip():
                    sharegpt_role = "tool" if role == "tool" else "human"
                    turns.append({"from": sharegpt_role, "value": text})
            elif isinstance(content, str) and content.strip():
                turns.append({"from": "human", "value": content})

        elif entry_type == "assistant":
            if isinstance(content, list):
                role, text = format_content_blocks(content)
                if text.strip():
                    turns.append({"from": "gpt", "value": text})
            elif isinstance(content, str) and content.strip():
                turns.append({"from": "gpt", "value": content})

    if not turns:
        return None

    # Merge consecutive same-role turns
    turns = merge_consecutive_turns(turns)

    # Filter out empty turns
    turns = [t for t in turns if t["value"].strip()]

    if not turns:
        return None

    return {"id": session_id, "conversations": turns}


def get_default_project_dir() -> str:
    """Get the project dir for the current working directory."""
    cwd = os.getcwd()
    sanitized = cwd.replace("/", "-")
    return os.path.expanduser(f"~/.claude/projects/{sanitized}")


def find_subagent_files(project_dir: str, session_id: str) -> list[str]:
    """Find subagent JSONL files for a given session."""
    subagent_dir = os.path.join(project_dir, session_id, "subagents")
    if not os.path.isdir(subagent_dir):
        return []
    return sorted(
        os.path.join(subagent_dir, f)
        for f in os.listdir(subagent_dir)
        if f.endswith(".jsonl")
    )


def find_session_files(project_dir: str, session_id: str | None = None) -> list[str]:
    """Find JSONL session files in the project directory."""
    if not os.path.isdir(project_dir):
        print(f"Error: project dir not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    files = []
    for f in sorted(os.listdir(project_dir)):
        if f.endswith(".jsonl"):
            if session_id is None or f.startswith(session_id):
                files.append(os.path.join(project_dir, f))
    return files


def convert_trajectory_file(path: str) -> list[dict]:
    """Convert the new trajectory.jsonl (api_request/api_response pairs) to ShareGPT.

    This format contains complete API interactions including system prompts,
    produced by the trajectoryLogger hook in query.ts.
    """
    entries = parse_jsonl(path)
    # Group by sessionId
    sessions: dict[str, list[dict]] = {}
    for e in entries:
        sid = e.get("sessionId", "unknown")
        sessions.setdefault(sid, []).append(e)

    results = []
    for sid, session_entries in sessions.items():
        # Sort by turnIndex
        session_entries.sort(key=lambda e: (e.get("turnIndex", 0), e.get("type", "")))

        turns = []
        system_prompt = None

        for entry in session_entries:
            if entry["type"] == "api_request":
                # Extract system prompt from first request
                if system_prompt is None and entry.get("system"):
                    sp = entry["system"]
                    if isinstance(sp, list):
                        system_prompt = "\n\n".join(
                            b.get("text", str(b)) if isinstance(b, dict) else str(b)
                            for b in sp
                        )
                    else:
                        system_prompt = str(sp)

                # Extract user/tool messages from the request
                messages = entry.get("messages", [])
                if not messages:
                    continue
                # Only take the last message (the new one in this turn)
                # Earlier messages are history already captured
                last_msg = messages[-1]
                role = last_msg.get("role", "")
                content = last_msg.get("content", "")

                if role == "user":
                    if isinstance(content, list):
                        fmt_role, text = format_content_blocks(content)
                        if text.strip():
                            sharegpt_role = "tool" if fmt_role == "tool" else "human"
                            turns.append({"from": sharegpt_role, "value": text})
                    elif isinstance(content, str) and content.strip():
                        turns.append({"from": "human", "value": content})

            elif entry["type"] == "api_response":
                content_blocks = entry.get("content", [])
                if content_blocks:
                    _, text = format_content_blocks(content_blocks)
                    if text.strip():
                        turns.append({"from": "gpt", "value": text})

        if not turns:
            continue

        # Add system prompt as first turn
        if system_prompt:
            turns.insert(0, {"from": "system", "value": system_prompt})

        turns = merge_consecutive_turns(turns)
        turns = [t for t in turns if t["value"].strip()]

        if turns:
            results.append({"id": sid, "conversations": turns})

    return results


def main():
    parser = argparse.ArgumentParser(description="Convert Claude Code JSONL transcripts to ShareGPT format")
    parser.add_argument("--input", "-i", help="Specific JSONL file to convert")
    parser.add_argument("--session", "-s", help="Session UUID to convert")
    parser.add_argument("--output", "-o", default="training_data.json", help="Output JSON path (default: training_data.json)")
    parser.add_argument("--project-dir", help="Project directory (default: auto-detect from cwd)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print output JSON")
    parser.add_argument("--trajectory", "-t", action="store_true",
                        help="Use trajectory.jsonl (full API request/response pairs with system prompt)")
    args = parser.parse_args()

    conversations = []

    # New trajectory.jsonl mode (complete API request/response pairs)
    if args.trajectory:
        project_dir = args.project_dir or get_default_project_dir()
        traj_path = os.path.join(project_dir, "trajectory.jsonl")
        if not os.path.isfile(traj_path):
            print(f"Error: trajectory.jsonl not found at {traj_path}", file=sys.stderr)
            print("Run conversations with the patched Claude Code first.", file=sys.stderr)
            sys.exit(1)
        print(f"Reading trajectory: {traj_path}", file=sys.stderr)
        conversations = convert_trajectory_file(traj_path)
        print(f"Found {len(conversations)} session(s) with system prompts", file=sys.stderr)

        output = {"conversations": conversations}
        indent = 2 if args.pretty else None
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=indent)
        total_turns = sum(len(c["conversations"]) for c in conversations)
        print(f"\nDone: {len(conversations)} sessions, {total_turns} total turns -> {args.output}", file=sys.stderr)
        return

    if args.input:
        # Single file mode
        files = [args.input]
    else:
        project_dir = args.project_dir or get_default_project_dir()
        print(f"Project dir: {project_dir}", file=sys.stderr)
        files = find_session_files(project_dir, args.session)

    if not files:
        print("No session files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(files)} session file(s)...", file=sys.stderr)

    for filepath in files:
        session_id = Path(filepath).stem
        print(f"  {session_id}...", file=sys.stderr, end=" ")

        entries = parse_jsonl(filepath)
        result = convert_session(entries, session_id)

        if result:
            conversations.append(result)
            n_turns = len(result["conversations"])
            print(f"{n_turns} turns", file=sys.stderr)
        else:
            print("skipped (empty)", file=sys.stderr)

        # Process subagent trajectories
        proj_dir = args.project_dir or get_default_project_dir() if not args.input else str(Path(filepath).parent)
        subagent_files = find_subagent_files(proj_dir, session_id)
        for sa_path in subagent_files:
            sa_name = Path(sa_path).stem
            sa_id = f"{session_id}/subagent/{sa_name}"
            print(f"    subagent {sa_name}...", file=sys.stderr, end=" ")

            sa_entries = parse_jsonl(sa_path)
            sa_result = convert_session(sa_entries, sa_id, is_subagent=True)

            if sa_result:
                conversations.append(sa_result)
                print(f"{len(sa_result['conversations'])} turns", file=sys.stderr)
            else:
                print("skipped (empty)", file=sys.stderr)

    output = {"conversations": conversations}

    indent = 2 if args.pretty else None
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=indent)

    total_turns = sum(len(c["conversations"]) for c in conversations)
    print(f"\nDone: {len(conversations)} sessions, {total_turns} total turns -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
