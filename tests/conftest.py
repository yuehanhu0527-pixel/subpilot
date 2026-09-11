"""pytest 共享 fixtures。"""

import os

# 在任何 app 模块 import 之前设置：data/schedule.json 存在时（如本地跑过
# demo 播种），ScheduleEngine 需要显式 tz——测试环境一律固定时区。
os.environ.setdefault("SUBPILOT_TIMEZONE", "America/New_York")

import pytest

from app.rag.store import ChromaStore


@pytest.fixture
def store():
    """每次测试一个独立的临时（内存）Chroma 实例。"""
    return ChromaStore.ephemeral()


@pytest.fixture
def collection(request):
    """每个测试唯一的 collection 名。

    Chroma 的 ephemeral 实例在同进程内共享底层状态，因此同名 collection
    会在测试之间串扰（例如维度冲突）。用测试名生成唯一名称来隔离。
    """
    name = "".join(ch if ch.isalnum() else "-" for ch in request.node.nodeid)
    return f"c-{name}"
