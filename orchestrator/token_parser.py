import re
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class EditFileAction:
    path: str
    content: str

@dataclass
class ClaimTaskAction:
    task_id: str

@dataclass
class CompleteTaskAction:
    task_id: str

@dataclass
class RunTestsAction:
    pass

@dataclass
class RunScriptAction:
    path: str

@dataclass
class ReadFileAction:
    path: str

# Lead Agent Actions
@dataclass
class AssignTaskAction:
    task_id: str
    assignee: str

@dataclass
class BroadcastAction:
    message: str

@dataclass
class RequestStatusAction:
    pass

@dataclass
class SynthesizeAction:
    summary: str

@dataclass
class ReleaseTaskAction:
    task_id: str

@dataclass
class CloseTaskAction:
    task_id: str

@dataclass
class VerifyTaskAction:
    task_id: str

# Planning phase (lead only)
@dataclass
class DiscoverTaskAction:
    id: str
    title: str
    description: str
    dependencies: List[str] = field(default_factory=list)  # list of task ids

# Parsing
EDIT_FILE_RE = re.compile(r'<edit_file\s+path="([^"]+)">\s*(.*?)\s*</edit_file>', re.DOTALL)
CLAIM_TASK_RE = re.compile(r'<claim_task\s+id="([^"]+)"\s*/>')
COMPLETE_TASK_RE = re.compile(r'<complete_task\s+id="([^"]+)"\s*/>')
RUN_TESTS_RE = re.compile(r'<run_tests\s*/>')
RUN_SCRIPT_RE = re.compile(r'<run_script\s+path="([^"]+)"\s*/>')
READ_FILE_RE  = re.compile(r'<read_file\s+path="([^"]+)"\s*/>')
ASSIGN_TASK_RE = re.compile(r'<assign_task\s+id="([^"]+)"\s+to="([^"]+)"\s*/>')
BROADCAST_RE = re.compile(r'<broadcast>\s*(.*?)\s*</broadcast>', re.DOTALL)
REQUEST_STATUS_RE = re.compile(r'<request_status\s*/>')
SYNTHESIZE_RE = re.compile(r'<synthesize>\s*(.*?)\s*</synthesize>', re.DOTALL)
RELEASE_TASK_RE = re.compile(r'<release_task\s+id="([^"]+)"\s*/>')
CLOSE_TASK_RE   = re.compile(r'<close_task\s+id="([^"]+)"\s*/>')
VERIFY_TASK_RE  = re.compile(r'<verify_task\s+id="([^"]+)"\s*/>')

_DISCOVER_TASK_PATTERN = (
    r'\s+id="([^"]+)"'
    r'\s+title="([^"]+)"'
    r'(?:\s+difficulty="[^"]*")?'  
    r'(?:\s+rel="[^"]*")?'          
    r'(?:\s+dependencies="([^"]*)")?'
    r'\s*>\s*(.*?)\s*'
)
DISCOVER_TASK_RE = re.compile(r'<discover_task' + _DISCOVER_TASK_PATTERN + r'</discover_task>', re.DOTALL)
PLAN_TASK_RE     = re.compile(r'<plan_task'     + _DISCOVER_TASK_PATTERN + r'</plan_task>',     re.DOTALL)

def _parse_discover_tasks(text):
    actions = []
    for m in list(DISCOVER_TASK_RE.finditer(text)) + list(PLAN_TASK_RE.finditer(text)):
        task_id     = m.group(1).strip()
        title       = m.group(2).strip()
        deps_raw    = m.group(3) or ""
        description = m.group(4).strip()
        dependencies = [d.strip() for d in deps_raw.split(",") if d.strip()]
        actions.append(DiscoverTaskAction(
            id=task_id,
            title=title,
            description=description,
            dependencies=dependencies,
        ))
    return actions


def parse_actions(text):
    """Parse action tags from agent response, preserving document order."""
    # Collect (start_pos, action) tuples so we can sort by position.
    positioned = []

    for m in EDIT_FILE_RE.finditer(text):
        positioned.append((m.start(), EditFileAction(path=m.group(1), content=m.group(2))))

    for m in CLAIM_TASK_RE.finditer(text):
        positioned.append((m.start(), ClaimTaskAction(task_id=m.group(1))))

    for m in COMPLETE_TASK_RE.finditer(text):
        positioned.append((m.start(), CompleteTaskAction(task_id=m.group(1))))

    # run_tests has no position anchor — collect first occurrence position if present
    m = RUN_TESTS_RE.search(text)
    if m:
        positioned.append((m.start(), RunTestsAction()))

    for m in RUN_SCRIPT_RE.finditer(text):
        positioned.append((m.start(), RunScriptAction(path=m.group(1))))

    for m in READ_FILE_RE.finditer(text):
        positioned.append((m.start(), ReadFileAction(path=m.group(1))))

    for m in ASSIGN_TASK_RE.finditer(text):
        positioned.append((m.start(), AssignTaskAction(task_id=m.group(1), assignee=m.group(2))))

    for m in BROADCAST_RE.finditer(text):
        positioned.append((m.start(), BroadcastAction(message=m.group(1))))

    m = REQUEST_STATUS_RE.search(text)
    if m:
        positioned.append((m.start(), RequestStatusAction()))

    for m in SYNTHESIZE_RE.finditer(text):
        positioned.append((m.start(), SynthesizeAction(summary=m.group(1))))

    for m in RELEASE_TASK_RE.finditer(text):
        positioned.append((m.start(), ReleaseTaskAction(task_id=m.group(1))))

    for m in CLOSE_TASK_RE.finditer(text):
        positioned.append((m.start(), CloseTaskAction(task_id=m.group(1))))

    for m in VERIFY_TASK_RE.finditer(text):
        positioned.append((m.start(), VerifyTaskAction(task_id=m.group(1))))

    for action in _parse_discover_tasks(text):
        # DiscoverTaskAction spans — find its position via id attribute
        id_pat = re.compile(r'<(?:discover_task|plan_task)[^>]*\bid="' + re.escape(action.id) + r'"')
        id_m = id_pat.search(text)
        pos = id_m.start() if id_m else len(text)
        positioned.append((pos, action))

    positioned.sort(key=lambda x: x[0])
    return [action for _, action in positioned]
