import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path.cwd().parent))
from astrbot_plugin_ob_memory.dashboard.server import DashboardServer
from astrbot_plugin_ob_memory.core.models import MemoryBucket

now = time.time()

class Manager:
    def __init__(self):
        self.buckets = [
            MemoryBucket(id='joy-high', session_id='demo', name='完成项目后的兴奋', content='我们聊到发布完成后的兴奋和期待。', valence=0.90, arousal=0.82, importance=7, activation_count=4, created_at=now, last_active_at=now, domain=['成长'], tags=['发布']),
            MemoryBucket(id='calm', session_id='demo', name='安静散步', content='傍晚散步让心情很平和。', valence=0.76, arousal=0.28, importance=4, activation_count=1, created_at=now-3600, last_active_at=now-3600, domain=['日常'], tags=['散步']),
            MemoryBucket(id='tense', session_id='demo', name='交付前压力', content='临近交付时有一点紧张。', valence=0.25, arousal=0.78, importance=6, activation_count=2, created_at=now-7200, last_active_at=now-7200, domain=['事务'], tags=['压力']),
            MemoryBucket(id='low', session_id='demo', name='疲惫的一天', content='今天状态比较低落疲惫。', valence=0.18, arousal=0.22, importance=3, activation_count=0, created_at=now-10800, last_active_at=now-10800, domain=['身心'], tags=['疲惫']),
        ]
    async def list_sessions(self):
        return ['demo']
    async def list_by_session(self, session_id, include_archived=False):
        return list(self.buckets)
    async def count_in_session(self, session_id):
        return {'event': len(self.buckets)}

plugin = SimpleNamespace(manager=Manager(), decay=None, embedding=None, config={})
server = DashboardServer(plugin, Path('.verify-dashboard-data-2141'))

async def main():
    await server.start(host='127.0.0.1', port=2141)
    while True:
        await asyncio.sleep(3600)

asyncio.run(main())
