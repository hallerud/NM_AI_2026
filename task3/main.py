
##### Del 1
import requests
 
BASE = "https://api.ainm.no"

# # Option 1: Cookie-based auth
# session = requests.Session()
# session.cookies.set("access_token", "YOUR_JWT_TOKEN")
 
# # Option 2: Bearer token auth
# session = requests.Session()
# session.headers["Authorization"] = "Bearer YOUR_JWT_TOKEN"


# ##### Del 2
# rounds = session.get(f"{BASE}/astar-island/rounds").json()
# active = next((r for r in rounds if r["status"] == "active"), None)
 
# if active:
#     round_id = active["id"]
#     print(f"Active round: {active['round_number']}")


# #### Del 3
# result = session.post(f"{BASE}/astar-island/simulate", json={
#     "round_id": round_id,
#     "seed_index": 0,
#     "viewport_x": 10,
#     "viewport_y": 5,
#     "viewport_w": 15,
#     "viewport_h": 15,
# }).json()

# grid = result["grid"]                # 15x15 terrain after simulation
# settlements = result["settlements"]  # settlements in viewport with full stats
# viewport = result["viewport"]        # {x, y, w, h}

# ##### Del 4
import numpy as np

# # TODO: DUMMY VARIABLES
# seeds = 30
# height = 0
# width = 0
# #######################

# for seed_idx in range(seeds):
#     prediction = np.full((height, width, 6), 1/6)  # uniform baseline
 
#     # TODO: replace with your model's predictions
#     # prediction[y][x] = [p_empty, p_settlement, p_port, p_ruin, p_forest, p_mountain]
 
#     resp = session.post(f"{BASE}/astar-island/submit", json={
#         "round_id": round_id,
#         "seed_index": seed_idx,
#         "prediction": prediction.tolist(),
#     })
#     print(f"Seed {seed_idx}: {resp.status_code}")


######## DUMMY HOST MAIN ######### 

# main.py
from fastapi import FastAPI
import uvicorn
import os

app = FastAPI()

@app.get("/health")
def health():
    """
    Health check endpoint for the competition validator.
    """
    return {"status": "ok"}

@app.post("/solve")
async def solve(request: dict):
   
    return 0

if __name__ == "__main__":
    # Get port from environment variable, default to 8080 for Cloud Run
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)






