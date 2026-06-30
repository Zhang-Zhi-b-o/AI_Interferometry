"""
Switch active model. Run: python select.py
Lists candidates, copies chosen one to current/model.pt
"""
import shutil
from pathlib import Path

SD = Path(__file__).parent
candidates = sorted((SD / "candidates").glob("*.pt"))
current = SD / "current" / "model.pt"

print("Candidates:")
for i, c in enumerate(candidates):
    marker = " ← CURRENT" if current.exists() and c.stat().st_size == current.stat().st_size else ""
    print(f"  [{i}] {c.name} ({c.stat().st_size/1e6:.1f}MB){marker}")

if not candidates:
    print("No candidate models found.")
    exit(1)

choice = input(f"\nSelect [0-{len(candidates)-1}]: ").strip()
try:
    idx = int(choice)
    if 0 <= idx < len(candidates):
        shutil.copy(candidates[idx], current)
        print(f"Done → current/model.pt = {candidates[idx].name}")
    else:
        print("Invalid index")
except ValueError:
    print("Enter a number")
