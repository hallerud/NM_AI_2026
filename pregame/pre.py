import asyncio
import websockets
 
async def play():
    async with websockets.connect("wss://game.ainm.no/ws?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJmNjE0ZTUwNy1lYzcwLTRkMjctOTJjNS1kNWExNDIwYzcxNjIiLCJ0ZWFtX2lkIjoiNjg1Njk3NTItNGNiMC00NWQzLTgzYzUtZWI0ZDNlOTg3NGVhIiwibWFwX2lkIjoiYzg5ZGEyZWMtM2NhNy00MGM5LWEzYjEtODAzNmZjYTNkMGI3IiwibWFwX3NlZWQiOjcwMDEsImRpZmZpY3VsdHkiOiJlYXN5IiwiZXhwIjoxNzczNTg5MDkzfQ.D4j7ZIPE7bxyTV-ff6mzPm6NMF1bTxDGFVnzeCRrfjY") as ws:
        while True:
            state = await ws.recv()
            # ... decide actions ...
            await ws.send('{"actions": [...]}')
 
asyncio.run(play())







