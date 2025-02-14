import asyncio

async def ping(host):
    process = await asyncio.create_subprocess_shell(
        f"ping -c 1 -W 1 {host}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await process.communicate()
    
    if process.returncode == 0:
        print(f"{host} is alive")

async def main():
    hosts = [f"ghc{i}.ghc.andrew.cmu.edu" for i in range(41, 88)]
    tasks = [ping(host) for host in hosts]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())