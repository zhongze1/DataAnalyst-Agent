"""旧数据迁移：把 ``data/uploads/*/meta.json`` 导入数据库。

**为什么需要**：改造前会话索引只存在 ``meta.json`` 里，进程重启靠扫目录恢复。
改造后元数据进了数据库，不迁移就会「丢失」历史会话（文件还在，但列表里没有）。

对话历史无法迁移 —— 之前从未持久化过，只可能从改造之后开始积累。

迁移在启动时自动执行（幂等），而不是提供手动脚本：手动脚本用户不会记得跑。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.storage.repository import SessionRecord
from app.storage.sqlite_repo import utc_now

logger = logging.getLogger(__name__)

META_NAME = "meta.json"


def needs_migration(repo, upload_dir: Path) -> bool:
    """判据：数据库里一个会话都没有，但磁盘上存在会话目录。"""
    if repo.count_sessions() > 0:
        return False
    if not upload_dir.exists():
        return False
    return any(
        folder.is_dir() and (folder / META_NAME).exists()
        for folder in upload_dir.iterdir()
    )


def migrate_meta_files(repo, upload_dir: Path) -> int:
    """把遗留 meta.json 导入数据库，返回导入条数。幂等。"""
    if not needs_migration(repo, upload_dir):
        return 0

    imported = 0
    for folder in sorted(upload_dir.iterdir()):
        meta_file = folder / META_NAME
        if not folder.is_dir() or not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("跳过无法解析的 meta.json: %s", meta_file)
            continue

        session_id = meta.get("session_id") or folder.name
        # 优先用 meta 里的文件名，找不到就退回目录里第一个文件
        filename = meta.get("filename")
        candidates = [
            p for p in sorted(folder.iterdir())
            if p.is_file() and p.name != META_NAME
        ]
        if not candidates:
            logger.warning("会话目录无文件，跳过: %s", folder)
            continue
        target = next((p for p in candidates if p.name == filename), candidates[0])
        if not target.exists():
            continue
        if repo.get_session(session_id) is not None:
            continue  # 幂等：已存在则跳过

        created = meta.get("created_at") or utc_now()
        repo.upsert_session(
            SessionRecord(
                session_id=session_id,
                filename=target.name,
                file_path=str(target),
                sheets=list(meta.get("sheets") or []),
                created_at=created,
                # 旧会话没有「最后活跃」概念，用创建时间兜底
                updated_at=created,
                turn_count=0,
            )
        )
        imported += 1

    if imported:
        logger.info("已从 meta.json 迁移 %d 个历史会话到数据库", imported)
    return imported
