# debug_poly.py
import asyncio
import aiohttp
import json

async def check():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://gamma-api.polymarket.com/markets?limit=50") as resp:
            data = await resp.json()
            print(f"Total markets fetched: {len(data)}")
            print("\nFirst 10 market questions:")
            for i, m in enumerate(data[:10]):
                print(f"{i+1}. {m.get('question', 'NO QUESTION')}")
                print(f"   Slug: {m.get('slug', '')}")
                print(f"   OutcomePrices: {m.get('outcomePrices', '')}\n")

if __name__ == "__main__":
    asyncio.run(check())