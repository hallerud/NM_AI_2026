#!/usr/bin/env python3
"""
download_history.py
===================
Downloads all completed rounds and their ground-truth data from the API.

Output: history/round_{id}_seed_{i}.json for every completed round + seed.

Each file contains:
  - round_id, round_number, seed_index
  - initial_grid       : H×W list of ints (initial terrain codes)
  - ground_truth       : H×W×6 list of floats (from Monte Carlo)
  - map_width, map_height

Run once before training:
    python download_history.py
"""

import requests
import json
import os
import time

BASE  = "https://api.ainm.no"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIzMzQwZWFlYS02ZjM0LTQ2ZDMtYjJjYi1lMjYxNmRkMDNhZjEiLCJlbWFpbCI6InNhbmRlcnR2ZXJyYUBnbWFpbC5jb20iLCJpc19hZG1pbiI6ZmFsc2UsImV4cCI6MTc3NDU0MzcxNn0.YEDTrEfdVpzA2uaHS0gztruqIP8hBf6H_qk255df0No"

session = requests.Session()
session.headers["Authorization"] = f"Bearer {TOKEN}"

OUTPUT_DIR = "history"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_rounds():
    return session.get(f"{BASE}/astar-island/rounds", timeout=30).json()


def get_analysis(round_id, seed_index):
    """Returns prediction + ground_truth + initial_grid for a completed seed."""
    r = session.get(f"{BASE}/astar-island/analysis/{round_id}/{seed_index}", timeout=30)
    if r.status_code == 200:
        return r.json()
    return None


def main():
    print("Fetching all rounds…")
    rounds = get_rounds()

    completed = [r for r in rounds if r["status"] in ("completed", "scoring")]
    print(f"Found {len(completed)} completed rounds (out of {len(rounds)} total)\n")

    total_saved = 0

    for r in completed:
        round_id     = r["id"]
        round_num    = r["round_number"]
        seeds_count  = r.get("seeds_count", 5)

        print(f"Round {round_num:3d}  id={round_id}")

        for seed_idx in range(seeds_count):
            out_path = os.path.join(OUTPUT_DIR, f"round_{round_num}_seed_{seed_idx}.json")

            if os.path.exists(out_path):
                print(f"  seed {seed_idx}: already downloaded, skipping")
                continue

            data = get_analysis(round_id, seed_idx)

            if data is None:
                print(f"  seed {seed_idx}: not available (round may not be fully scored)")
                continue

            # Save only what we need for training
            record = {
                "round_id":    round_id,
                "round_number": round_num,
                "seed_index":  seed_idx,
                "map_width":   data["width"],
                "map_height":  data["height"],
                "initial_grid": data["initial_grid"],
                "ground_truth": data["ground_truth"],   # H×W×6 probabilities
            }

            with open(out_path, "w") as f:
                json.dump(record, f)

            print(f"  seed {seed_idx}: saved  ({data['width']}×{data['height']})")
            total_saved += 1
            time.sleep(0.15)  # be polite to the API

    print(f"\nDone. {total_saved} new files saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
