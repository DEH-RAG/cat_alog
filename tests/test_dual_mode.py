"""Dual-mode, end-to-end proof of the CATALOG catalogue cards on the real app.

Moved out of the CAT core (tests/catalog/test_dual_mode.py) and owned by the
cat_alog plugin itself, exactly like test_catalog_phase_integration.py.

Standalone script (plugin rules): the host imports every ``.py`` under the
plugin folder, so the ENTIRE body lives inside ``if __name__ == "__main__":``
and importing this file executes zero code. Run it from the host checkout:

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat_alog/tests/test_dual_mode.py

Isolation
---------
* Redis: forced to the dedicated ``CAT_REDIS_DB=15`` BEFORE ``cat`` is imported;
  every operation aborts if the bound db is not 15. This script NEVER calls
  ``flushdb``/``flushall``: it resets only its own namespaces
  (``system:*`` and ``agents:catalog_dual_*``) through ``cat.db.crud``.
* Qdrant: a fresh in-memory client per scenario (same double as the old host
  conftest).
* Neo4j: the live instance at ``bolt://127.0.0.1:7687``; isolation is a unique
  tenant id (``catalog_dual_<uuid>``) and every node carrying that tenant is
  deleted in ``finally``. If Neo4j is unreachable the run FAILS explicitly
  (never skips silently).

The same real corpus (``tests/mocks/sample.txt``) is ingested through both
ingestion engines on the real app (``create_app`` + ``LifespanManager``):

* **classic mode** — the system ``ingestion`` setting is the core engine, so the
  cat_alog ``before_rabbithole_splits_documents`` / ``before_rabbithole_stores_documents``
  hooks own the card;
* **efficient mode** — the system ``ingestion`` setting is
  ``EfficientIngestionConfiguration``, so the ``catalog_cards`` phase (declared
  by cat_alog through the EffING phase contract) owns the card and the classic
  hooks are never fired.

The old pytest fixtures are inlined below (a standalone script cannot use
fixtures): the plugin symlinks, the MyGRAPH link, the in-memory Qdrant double,
the plugin-manager/background-task doubles and the real app boot all happen in
``_scenario_env`` with the same setup/teardown semantics, except the Redis
cleanup is scoped instead of a flush.
"""

if __name__ == "__main__":

    import asyncio
    import functools
    import importlib
    import json
    import os
    import shutil
    import subprocess
    import sys
    import time
    import traceback
    import uuid
    from contextlib import asynccontextmanager
    from inspect import iscoroutinefunction
    from pathlib import Path

    PLUGINS_ROOT = Path("/root/git/plugins")

    #: The plugin root: this file lives in ``tests/`` after the move.
    REPO_ROOT = Path(__file__).resolve().parent.parent

    # effing_testkit is published by the EffING plugin: the shared, import-safe
    # doubles for the phase contract consumed by cat_alog. It now lives in the
    # plugin's ``tests/`` directory.
    _effing_dir = str(PLUGINS_ROOT / "cat-efficient-ingestion" / "tests")
    if _effing_dir not in sys.path:
        sys.path.insert(0, _effing_dir)
    import effing_testkit

    # ---------------------------------------------------------------------------
    # Environment BEFORE importing cat (cat.startup builds an app at import time).
    # Redis: dedicated db 15, NEVER production. No flushdb/flushall is used.
    # ---------------------------------------------------------------------------
    TEST_REDIS_DB = "15"


    def _resolve_redis_host():
        explicit = os.environ.get("CAT_REDIS_HOST")
        if explicit:
            return explicit
        for name in ("grinning_cat_redis", "dev-grinning-cat-redis-1", "redis"):
            try:
                out = subprocess.run(
                    ["docker", "inspect", "-f",
                     "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name],
                    capture_output=True, text=True, timeout=5,
                )
            except Exception:  # noqa: BLE001 - fall through to localhost
                continue
            ip = out.stdout.strip()
            if out.returncode == 0 and ip:
                return ip
        return "127.0.0.1"


    os.environ.setdefault("CAT_JWT_SECRET", "test_only_jwt_secret_0123456789_abcdefghij")
    os.environ["CAT_REDIS_DB"] = TEST_REDIS_DB
    os.environ.setdefault("CAT_REDIS_HOST", _resolve_redis_host())
    os.environ.setdefault("CAT_REDIS_PORT", "6379")
    os.environ.setdefault("CAT_QDRANT_HOST", "grinning_cat_vector_memory")

    from asgi_lifespan import LifespanManager
    from httpx import ASGITransport, AsyncClient
    from neo4j import AsyncGraphDatabase
    from qdrant_client import AsyncQdrantClient

    from cat import utils
    from cat.auth import auth_utils
    from cat.auth.permissions import AuthPermission, AuthResource
    from cat.db import crud
    from cat.db.database import get_sync_db
    import cat.db.database as db_module
    from cat.db.cruds import conversations as crud_conversations
    from cat.looking_glass.mad_hatter.plugin import Plugin
    from cat.routes import routes_utils
    from cat.services.factory.embedder import Embeddings
    from cat.services.factory.vector_db import QdrantHandler
    from cat.services.memory.models import PointStruct, VectorMemoryType
    from cat.startup import create_app

    # Host test utilities inlined from tests.utils (standalone script).
    api_key = "meow_http"
    jwt_secret = "meow_jwt_test_secret_at_least_32_characters"


    async def create_new_user(client, username="Alice", headers=None, permissions=None, password=None):
        new_user = {
            "username": username,
            "password": password or "wandering_in_wonderland",
            "permissions": permissions or {str(AuthResource.CHAT): [str(AuthPermission.WRITE)]},
        }
        response = await client.post("/users/", json=new_user, headers=headers)
        assert response.status_code == 200, response.text
        return response.json()


    # ---------------------------------------------------------------------------
    # Constants (from the original suite)
    # ---------------------------------------------------------------------------
    PLUGIN_SRC = Path("/root/git/plugins/cat-graphrag.my")
    PLUGIN_ID = "cat_graphrag"
    PLUGINS_DIR = Path(utils.get_base_path()) / "plugins"
    PLUGIN_LINK = PLUGINS_DIR / PLUGIN_ID

    CORE_PLUGINS_DIR = os.path.join("cat", "core_plugins")
    PLUGIN_LINKS = {
        "cat_alog": str(REPO_ROOT),
        "cat_efficient_ingestion": str(PLUGINS_ROOT / "cat-efficient-ingestion"),
        "cat_multimodal_ingestion": str(PLUGINS_ROOT / "cat-multimodal-ingestion"),
    }

    #: links created by this run; must be gone at the end (cleanup receipt)
    CREATED_LINKS: list[str] = []
    #: agents created by this run; their namespaces are the only ones cleaned
    CREATED_AGENTS: set[str] = set()
    SCOPE_PREFIX = "catalog_dual_"

    NEO4J_URI = "bolt://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_DATABASE = "neo4j"
    NEO4J_AUTH_CONTAINERS = ("neo4j-db", "minimal-neo4j-1", "dev-neo4j-1")

    SAMPLE_FILE = Path("tests/mocks/sample.txt")
    SAMPLE_NAME = SAMPLE_FILE.name
    SAMPLE_CONTENT_TYPE = "text/plain"
    LONG_NAME = "long_catalog.txt"

    CORE_ENGINE = "CoreIngestionConfiguration"
    EFFICIENT_ENGINE = "EfficientIngestionConfiguration"

    DECLARATIVE = str(VectorMemoryType.DECLARATIVE)
    EPISODIC = str(VectorMemoryType.EPISODIC)

    # The suite expects to run from the host checkout (sample corpus + file storage).
    if not SAMPLE_FILE.exists():
        _core_root = Path("/root/git/grinning-cat-core")
        if (_core_root / SAMPLE_FILE).exists():
            os.chdir(_core_root)


    # ---------------------------------------------------------------------------
    # Minimal pytest-monkeypatch equivalent (setattr + undo)
    # ---------------------------------------------------------------------------
    _UNSET = object()


    class _MonkeyPatch:
        def __init__(self):
            self._undo = []

        def setattr(self, target, name, value=None, raising=True):
            old = getattr(target, name, _UNSET)
            if old is _UNSET and raising:
                raise AttributeError(f"{target!r} has no attribute {name!r}")
            self._undo.append((target, name, old))
            setattr(target, name, value)

        def undo(self):
            while self._undo:
                target, name, old = self._undo.pop()
                if old is _UNSET:
                    try:
                        delattr(target, name)
                    except AttributeError:
                        pass
                else:
                    setattr(target, name, old)


    # ---------------------------------------------------------------------------
    # Host test doubles + inlined fixtures
    # ---------------------------------------------------------------------------
    def _install_host_doubles(mp, memory_client):
        def _mock_qdrant_init(self, *args, **kwargs):
            super(QdrantHandler, self).__init__()
            self._client = memory_client
            self.save_memory_snapshots = kwargs.get("save_memory_snapshots", False)

        mp.setattr(QdrantHandler, "__init__", _mock_qdrant_init)
        mp.setattr(QdrantHandler, "close", lambda self: None)

        def _mock_get_redis_kwargs():
            return {
                "host": os.environ["CAT_REDIS_HOST"],
                "port": int(os.environ["CAT_REDIS_PORT"]),
                "db": TEST_REDIS_DB,
                "encoding": "utf-8",
                "decode_responses": True,
            }

        mp.setattr(db_module, "get_redis_kwargs", _mock_get_redis_kwargs)
        mp.setattr(utils, "get_plugins_path", lambda: "tests/mocks/mock_plugin_folder/")
        mp.setattr(utils, "get_file_manager_root_storage_path", lambda: "tests/data/storage")

        async def _noop_install_requirements(self):
            return None

        mp.setattr(Plugin, "_install_requirements", _noop_install_requirements)
        mp.setattr(auth_utils, "extract_agent_id_from_request", lambda *a, **k: "agent_test")

        async def _immediate_task(background_tasks, func, *args, **kwargs):
            if iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)

        mp.setattr(routes_utils, "run_background_task", _immediate_task)


    def _create_catalog_plugin_symlinks():
        for plugin_id, target in PLUGIN_LINKS.items():
            link = os.path.join(CORE_PLUGINS_DIR, plugin_id)
            if not os.path.lexists(link):
                os.symlink(target, link)
                CREATED_LINKS.append(link)


    def _remove_catalog_plugin_symlinks():
        for link in list(CREATED_LINKS):
            if os.path.lexists(link):
                os.remove(link)
            if link in CREATED_LINKS:
                CREATED_LINKS.remove(link)


    def _ensure_graph_link() -> bool:
        created = False
        if PLUGIN_LINK.exists() or PLUGIN_LINK.is_symlink():
            target = os.readlink(PLUGIN_LINK) if PLUGIN_LINK.is_symlink() else str(PLUGIN_LINK)
            if Path(target).resolve() != PLUGIN_SRC.resolve():
                raise AssertionError(f"{PLUGIN_LINK} already exists and points to {target}")
        else:
            PLUGIN_LINK.symlink_to(PLUGIN_SRC, target_is_directory=True)
            created = True
        return created


    def _remove_graph_link(created: bool):
        if created and PLUGIN_LINK.is_symlink() and Path(os.readlink(PLUGIN_LINK)) == PLUGIN_SRC:
            PLUGIN_LINK.unlink()
        assert not (created and (PLUGIN_LINK.exists() or PLUGIN_LINK.is_symlink()))


    def _bound_db() -> str:
        db = get_sync_db()
        return str(db.connection_pool.connection_kwargs.get("db", os.environ.get("CAT_REDIS_DB")))


    def _assert_test_db() -> None:
        bound = _bound_db()
        if bound != TEST_REDIS_DB:
            raise RuntimeError(
                f"Refusing to run: bound Redis db={bound!r}, expected the dedicated test db={TEST_REDIS_DB!r}."
            )


    async def _reset_scoped_state() -> None:
        # Scoped cleanup only: this suite's namespaces, never FLUSHDB/FLUSHALL.
        _assert_test_db()
        await crud.destroy("system:*")
        await crud.destroy(f"agents:{SCOPE_PREFIX}*")


    def _remove_storage_dirs():
        storage = Path("tests/data/storage")
        if not storage.exists():
            return
        for child in storage.glob(f"{SCOPE_PREFIX}*"):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        # prune only the directories this run may have created (never a populated one)
        for parent in (storage, Path("tests/data")):
            try:
                parent.rmdir()
            except OSError:
                pass


    @asynccontextmanager
    async def _scenario_env():
        """Inlined conftest + fixtures: one fresh app, same setup/teardown order."""
        mp = _MonkeyPatch()
        memory_client = AsyncQdrantClient(":memory:")
        created_graph = False
        app = None
        previous_api_key = os.environ.get("CAT_API_KEY")
        previous_jwt = os.environ.get("CAT_JWT_SECRET")
        try:
            utils.singleton.instances.clear()
            _install_host_doubles(mp, memory_client)

            _create_catalog_plugin_symlinks()
            created_graph = _ensure_graph_link()
            # restore the real plugins folder so MyGRAPH is discovered by the app
            mp.setattr(utils, "get_plugins_path", lambda: str(PLUGINS_DIR))

            # EffING contract (import-safe testkit): the dual-mode suite needs the
            # REAL phase machine, so the canonical package must be importable here.
            effing_mode = effing_testkit.install_stub_package()
            assert effing_mode == "real", (
                "the dual-mode suite needs the real cat_efficient_ingestion plugin importable, "
                f"got effing_testkit mode={effing_mode!r}"
            )

            os.environ["CAT_API_KEY"] = api_key
            os.environ["CAT_JWT_SECRET"] = jwt_secret

            await _reset_scoped_state()

            app = create_app()
            async with LifespanManager(app) as manager:
                async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac:
                    ac._fastapi_test_app = app
                    _assert_test_db()
                    yield ac, mp
        finally:
            if app is not None:
                assert not hasattr(app.state, "lizard")
            if previous_api_key is None:
                os.environ.pop("CAT_API_KEY", None)
            else:
                os.environ["CAT_API_KEY"] = previous_api_key
            if previous_jwt is None:
                os.environ.pop("CAT_JWT_SECRET", None)
            else:
                os.environ["CAT_JWT_SECRET"] = previous_jwt
            _remove_graph_link(created_graph)
            _remove_catalog_plugin_symlinks()
            effing_testkit.uninstall_stub_package()
            _remove_storage_dirs()
            try:
                await _reset_scoped_state()
            finally:
                collections = await memory_client.get_collections()
                for collection in collections.collections:
                    await memory_client.delete_collection(collection.name)
                await memory_client.close()
                mp.undo()
                utils.singleton.instances.clear()


    async def _run_scenario(name, func):
        agent_key = f"{SCOPE_PREFIX}{uuid.uuid4().hex[:10]}"
        CREATED_AGENTS.add(agent_key)
        async with _scenario_env() as (client, mp):
            await func(client, agent_key, mp)

    # ---------------------------------------------------------------------------
    # Neo4j helpers (real instance; tenant-scoped cleanup in finally)
    # ---------------------------------------------------------------------------
    def _neo4j_password() -> str:
        password = os.environ.get("CAT_NEO4J_PASSWORD") or os.environ.get("NEO4J_PASSWORD")
        if password:
            return password

        import subprocess

        for container in NEO4J_AUTH_CONTAINERS:
            try:
                proc = subprocess.run(
                    ["docker", "inspect", container, "--format", "{{json .Config.Env}}"],
                    capture_output=True, text=True, timeout=10,
                )
            except Exception:  # noqa: BLE001 - try the next container
                continue
            if proc.returncode != 0:
                continue
            try:
                env = json.loads(proc.stdout)
            except json.JSONDecodeError:
                continue
            for entry in env:
                if entry.startswith("NEO4J_AUTH="):
                    _, _, password = entry.partition("=")[2].partition("/")
                    if password:
                        return password

        raise RuntimeError(
            "Neo4j password not found: set CAT_NEO4J_PASSWORD/NEO4J_PASSWORD or run "
            f"one of the local containers {NEO4J_AUTH_CONTAINERS} with NEO4J_AUTH."
        )


    def _neo4j_driver():
        return AsyncGraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, _neo4j_password()), connection_timeout=10,
        )


    async def _neo4j_counts(tenant: str) -> dict:
        driver = _neo4j_driver()
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:

                async def one(query):
                    result = await session.run(query, tenant=tenant)
                    record = await result.single()
                    await result.consume()
                    return record["c"] if record else 0

                return {
                    "nodes": await one("MATCH (n {tenant_id: $tenant}) RETURN count(n) AS c"),
                    "relationships": await one(
                        "MATCH (a {tenant_id: $tenant})-[r]->(b {tenant_id: $tenant}) RETURN count(r) AS c"
                    ),
                    "documents": await one("MATCH (d:Document {tenant_id: $tenant}) RETURN count(d) AS c"),
                    "cards": await one(
                        "MATCH (d:Document {tenant_id: $tenant}) "
                        "WHERE d.metadata CONTAINS '\"is_catalogue_card\": true' RETURN count(d) AS c"
                    ),
                }
        finally:
            await driver.close()


    async def _wipe_tenant(tenant: str) -> dict:
        driver = _neo4j_driver()
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                result = await session.run("MATCH (n {tenant_id: $tenant}) DETACH DELETE n", tenant=tenant)
                await result.consume()
        finally:
            await driver.close()
        return await _neo4j_counts(tenant)


    async def _assert_neo4j_reachable() -> None:
        driver = _neo4j_driver()
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                result = await session.run("RETURN 1 AS x")
                await result.single()
                await result.consume()
        except Exception as exc:  # noqa: BLE001 - surfaced explicitly, never silently skipped
            raise AssertionError(f"Neo4j is unreachable at {NEO4J_URI}: {exc!r}")
        finally:
            await driver.close()


    async def _next_pairs(tenant: str, collection: str = DECLARATIVE) -> list[tuple[str, str, dict, dict]]:
        """Every ``NEXT`` edge of the tenant in ``collection``, with endpoint metadata."""
        driver = _neo4j_driver()
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                result = await session.run(
                    "MATCH (a:Document {tenant_id: $tenant})-[:BELONGS_TO]->"
                    "(c:Collection {tenant_id: $tenant, name: $collection}) "
                    "MATCH (a)-[r:NEXT]->(b:Document {tenant_id: $tenant}) "
                    "RETURN a.id AS aid, b.id AS bid, a.metadata AS am, b.metadata AS bm",
                    tenant=tenant, collection=collection,
                )
                rows = [record async for record in result]
                await result.consume()
        finally:
            await driver.close()

        def _meta(raw):
            try:
                return json.loads(raw) if raw else {}
            except (TypeError, json.JSONDecodeError):
                return {}

        return [(r["aid"], r["bid"], _meta(r["am"]), _meta(r["bm"])) for r in rows]


    async def _count_tenant_prefix(prefix: str) -> int:
        driver = _neo4j_driver()
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                result = await session.run(
                    "MATCH (n) WHERE n.tenant_id STARTS WITH $prefix RETURN count(n) AS c",
                    prefix=prefix,
                )
                record = await result.single()
                await result.consume()
                return record["c"] if record else 0
        finally:
            await driver.close()


    # ---------------------------------------------------------------------------
    # App-level configuration helpers
    # ---------------------------------------------------------------------------
    def _headers(agent_key: str) -> dict:
        return {"X-Agent-ID": agent_key, "Authorization": f"Bearer {api_key}"}


    def _neo4j_payload() -> dict:
        # The derived graph (SourceFile/PART_OF/NEXT) is always built; the heavy
        # entity/concept extraction is disabled so the test isolates the chunk
        # structure and stays fast (and offline).
        return {
            "neo4j_uri": NEO4J_URI,
            "neo4j_user": NEO4J_USER,
            "neo4j_password": _neo4j_password(),
            "neo4j_database": NEO4J_DATABASE,
            "enable_derived_graph": True,
            "enable_entity_extraction": False,
            "spacy_models": {"en": "en_core_web_sm"},
            "enable_knowledge_graph": False,
            "enable_concept_relations": False,
        }


    async def _activate_plugin(lizard, plugin_id: str) -> None:
        if plugin_id not in lizard.plugin_manager.active_plugins:
            await lizard.plugin_manager.activate_plugin(plugin_id)


    async def _set_neo4j(client, agent_key: str) -> None:
        response = await client.put(
            "/vector_database/settings/Neo4jGraphRAGConfig",
            json=_neo4j_payload(),
            headers=_headers(agent_key),
        )
        assert response.status_code == 200, response.text
        assert response.json()["name"] == "Neo4jGraphRAGConfig"


    async def _set_file_manager(client, agent_key: str) -> None:
        response = await client.put(
            "/file_manager/settings/LocalFileManagerConfig", json={}, headers=_headers(agent_key),
        )
        assert response.status_code == 200, response.text


    async def _set_engine(client, agent_key: str, ingestion_name: str) -> None:
        response = await client.put(
            f"/ingestion/settings/{ingestion_name}", json={}, headers=_headers(agent_key),
        )
        assert response.status_code == 200, response.text


    async def _configure_agent(client, agent_key: str, ingestion_name: str, graphrag: bool = False):
        CREATED_AGENTS.add(agent_key)

        lizard = client._fastapi_test_app.state.lizard
        if graphrag:
            await _activate_plugin(lizard, PLUGIN_ID)
        cat = await lizard.create_cheshire_cat(agent_key)
        if graphrag:
            if PLUGIN_ID not in cat.plugin_manager.active_plugins:
                await cat.plugin_manager.activate_plugin(PLUGIN_ID)
            assert PLUGIN_ID in cat.plugin_manager.active_plugins, cat.plugin_manager.active_plugins
            await _set_neo4j(client, agent_key)
        await _set_engine(client, agent_key, ingestion_name)
        # the upstream default file manager is a no-op; the prompt list and the
        # oversized-card abstract need the real local storage
        await _set_file_manager(client, agent_key)
        cat = await lizard.get_cheshire_cat(agent_key)
        if graphrag:
            assert type(cat.vector_memory_handler).__name__ == "GraphRAGHandler", type(cat.vector_memory_handler)
        return cat


    async def _ingest(client, agent_key: str, name: str, data: bytes, content_type: str = "text/plain", chat_id=None):
        url = "/rabbithole/" if not chat_id else f"/rabbithole/{chat_id}"
        return await client.post(url, files={"file": (name, data, content_type)}, headers=_headers(agent_key))


    async def _ingest_sample(client, agent_key: str, chat_id=None):
        return await _ingest(client, agent_key, SAMPLE_NAME, SAMPLE_FILE.read_bytes(), SAMPLE_CONTENT_TYPE, chat_id)


    async def _status(agent_key: str, source: str = SAMPLE_NAME, scope: str = "agent") -> dict:
        registry = importlib.import_module("cat.core_plugins.cat_efficient_ingestion.registry")
        return await registry.get_status(agent_key, scope, source) or {}


    # ---------------------------------------------------------------------------
    # Point/assertion helpers
    # ---------------------------------------------------------------------------
    async def _points(cat, collection: str, metadata: dict) -> list:
        records, _ = await cat.vector_memory_handler.get_all_tenant_points(
            collection, metadata=metadata, with_vectors=False,
        )
        return list(records or [])


    async def _card_points(cat, collection: str, source: str) -> list:
        return await _points(cat, collection, {"source": source, "is_catalogue_card": True})


    async def _chunk_points(cat, collection: str, source: str) -> list:
        records = await _points(cat, collection, {"source": source})
        return [
            r for r in records
            if not (r.payload or {}).get("metadata", {}).get("is_catalogue_card")
            and (r.payload or {}).get("metadata", {}).get("image") is not True
        ]


    def _meta(record) -> dict:
        return (record.payload or {}).get("metadata", {}) or {}


    def _norm_id(point_id) -> str:
        """Normalize a point id for comparison across UUID representations.

        The product mints ``uuid5(...).hex`` (undashed); Qdrant returns stored point
        ids in the canonical dashed UUID form. Compare the identity, not the format.
        """
        return str(point_id).replace("-", "").lower()


    def _order_key(record) -> tuple:
        meta = _meta(record)
        return (meta.get("start_index", 1 << 60), meta.get("chunk_index", 1 << 60))


    async def _assert_next_chain(agent_key: str, chunks: list) -> list:
        """Assert the ``NEXT`` edges are exactly the start_index-ordered chain."""
        ordered = sorted(chunks, key=_order_key)
        pairs = await _next_pairs(agent_key)
        actual = {(a, b) for a, b, _, _ in pairs}
        expected = {(ordered[i].id, ordered[i + 1].id) for i in range(len(ordered) - 1)}
        assert actual == expected, (actual, expected)
        for _, _, a_meta, b_meta in pairs:
            assert b_meta["start_index"] > a_meta["start_index"], (a_meta, b_meta)
        return ordered


    async def _prompt_list(cat, base: str = "You are a helpful cat.") -> str:
        hook = next(
            h for h in cat.plugin_manager.plugins["cat_alog"].hooks if h.name == "agent_prompt_suffix"
        )
        return await hook.function(base, cat)


    def _cat_alog_module(cat):
        hook = next(
            h for h in cat.plugin_manager.plugins["cat_alog"].hooks if h.name == "agent_prompt_suffix"
        )
        return importlib.import_module(hook.function.__module__)


    def _install_card_hook_spy(plugin_manager, mp) -> dict:
        """Count the cat_alog classic card work (test-only instrumentation).

        ``stores`` counts the hook that APPENDS the classic card; ``cache_writes``
        counts the classic summary-cache writes (the only card work the splits hook
        does). EffING's shared chunking step still invokes the splits hook, but with
        the phase machine active it must be a no-op: no summary, no cache write.
        """
        counts = {"splits": 0, "stores": 0, "cache_writes": 0}
        wanted = {
            "before_rabbithole_splits_documents": "splits",
            "before_rabbithole_stores_documents": "stores",
        }

        def _wrap(key, original):
            @functools.wraps(original)
            async def wrapper(*args, **kwargs):
                counts[key] += 1
                return await original(*args, **kwargs)

            return wrapper

        # resolve the plugin module through an untouched hook (the wrapped ones
        # would report this test module as their __module__ after monkeypatching)
        cat_alog_module = None
        for hook in plugin_manager.plugins["cat_alog"].hooks:
            if hook.name == "agent_prompt_suffix":
                cat_alog_module = importlib.import_module(hook.function.__module__)
                break

        for name, key in wanted.items():
            for hook in plugin_manager.hooks.get(name, []):
                if hook.plugin_id == "cat_alog":
                    mp.setattr(hook, "function", _wrap(key, hook.function))

        assert cat_alog_module is not None, "cat_alog hooks not registered"
        original_set = cat_alog_module.CATALOGUES.set

        def _counting_set(*args, **kwargs):
            counts["cache_writes"] += 1
            return original_set(*args, **kwargs)

        mp.setattr(cat_alog_module.CATALOGUES, "set", _counting_set)
        assert counts == {"splits": 0, "stores": 0, "cache_writes": 0}
        return counts



    # ---------------------------------------------------------------------------
    # Scenario 1 — classic mode
    # ---------------------------------------------------------------------------
    async def _scenario_classic_mode_one_card_ordered_next_and_prompt_once(client, agent_key, mp):
        await _assert_neo4j_reachable()
        try:
            cat = await _configure_agent(client, agent_key, CORE_ENGINE, graphrag=True)
            spy = _install_card_hook_spy(cat.plugin_manager, mp)

            response = await _ingest_sample(client, agent_key)
            assert response.status_code == 200, response.text

            cards = await _card_points(cat, DECLARATIVE, SAMPLE_NAME)
            assert len(cards) == 1, [(_meta(c), c.id) for c in cards]
            assert _meta(cards[0])["is_catalogue_card"] is True

            chunks = await _chunk_points(cat, DECLARATIVE, SAMPLE_NAME)
            assert len(chunks) >= 2, len(chunks)
            for chunk in chunks:
                meta = _meta(chunk)
                assert isinstance(meta.get("chunk_index"), int), meta
                assert isinstance(meta.get("start_index"), int), meta

            ordered = await _assert_next_chain(agent_key, chunks)
            assert len(await _next_pairs(agent_key)) == len(chunks) - 1

            prompt = await _prompt_list(cat)
            assert prompt.count(SAMPLE_NAME) == 1, prompt

            # classic mode owns the card through the two classic hooks, exactly once
            assert spy == {"splits": 1, "stores": 1, "cache_writes": 1}, spy

            print("CLASSIC_CATALOG", {
                "cards": len(cards), "chunks": len(chunks),
                "next": len(await _next_pairs(agent_key)),
                "starts": [_meta(c)["start_index"] for c in ordered], "spy": spy,
            })
        finally:
            leftover = await _wipe_tenant(agent_key)
            assert leftover == {"nodes": 0, "relationships": 0, "documents": 0, "cards": 0}, leftover


    # ---------------------------------------------------------------------------
    # Scenario 2 — efficient mode
    # ---------------------------------------------------------------------------
    async def _scenario_efficient_mode_phase_builds_one_card_and_never_fires_classic_hooks(client, agent_key, mp):
        await _assert_neo4j_reachable()
        try:
            cat = await _configure_agent(client, agent_key, EFFICIENT_ENGINE, graphrag=True)
            spy = _install_card_hook_spy(cat.plugin_manager, mp)

            response = await _ingest_sample(client, agent_key)
            assert response.status_code == 200, response.text

            status = await _status(agent_key)
            assert status.get("status") == "completed", status
            recorded = set(status.get("completed_phases") or {})
            assert {"parsing_chunking", "embedding", "catalog_cards"} <= recorded, recorded

            cards = await _card_points(cat, DECLARATIVE, SAMPLE_NAME)
            assert len(cards) == 1, [(_meta(c), c.id) for c in cards]
            assert _meta(cards[0])["is_catalogue_card"] is True

            # the classic card work is inert: the phase owns the card. EffING emits
            # the store hook once (both engines emit it exactly once per source),
            # but the phase machine makes it a no-op: no card work and no summary
            # cache write.
            assert spy["stores"] == 1 and spy["cache_writes"] == 0, spy

            chunks = await _chunk_points(cat, DECLARATIVE, SAMPLE_NAME)
            assert len(chunks) >= 2, len(chunks)
            ordered = await _assert_next_chain(agent_key, chunks)

            prompt = await _prompt_list(cat)
            assert f"**{SAMPLE_NAME}**" in prompt, prompt
            assert prompt.count(SAMPLE_NAME) == 1, prompt

            print("EFFICIENT_CATALOG", {
                "cards": len(cards), "chunks": len(chunks),
                "phases": sorted(recorded),
                "starts": [_meta(c)["start_index"] for c in ordered], "spy": spy,
            })
        finally:
            leftover = await _wipe_tenant(agent_key)
            assert leftover == {"nodes": 0, "relationships": 0, "documents": 0, "cards": 0}, leftover


    # ---------------------------------------------------------------------------
    # Scenario 3 — EPISODIC chat-scoped card
    # ---------------------------------------------------------------------------
    async def _scenario_episodic_card_is_chat_scoped(client, agent_key, mp):
        chat = f"catalog_chat_{uuid.uuid4().hex[:8]}"
        cat = await _configure_agent(client, agent_key, CORE_ENGINE)

        response = await _ingest_sample(client, agent_key, chat_id=chat)
        assert response.status_code == 200, response.text

        cards = await _card_points(cat, EPISODIC, SAMPLE_NAME)
        assert len(cards) == 1, [(_meta(c), c.id) for c in cards]
        for card in cards:
            assert _meta(card).get("chat_id") == chat, _meta(card)
        assert await _card_points(cat, DECLARATIVE, SAMPLE_NAME) == []

        print("EPISODIC_CATALOG", {"segments": len(cards), "chat_id": chat})


    # ---------------------------------------------------------------------------
    # Scenario 4 — oversized card is segmented, every segment carries full_card
    # ---------------------------------------------------------------------------
    async def _scenario_oversized_card_is_segmented_with_full_card(client, agent_key, mp):
        # test-only instrumentation: lower the embedder input window so the card
        # must be split into several segments (DumbEmbedder normally has no limit)
        mp.setattr(Embeddings, "max_input_tokens", 100)

        cat = await _configure_agent(client, agent_key, EFFICIENT_ENGINE)
        long_text = ("Alice was beginning to get very tired of sitting by her sister. " * 220).encode()

        response = await _ingest(client, agent_key, LONG_NAME, long_text)
        assert response.status_code == 200, response.text

        cards = await _card_points(cat, DECLARATIVE, LONG_NAME)
        assert len(cards) > 1, len(cards)
        full_cards = {_meta(c).get("full_card") for c in cards}
        assert len(full_cards) == 1, len(full_cards)
        full_card = full_cards.pop()
        assert isinstance(full_card, str) and "## Summary" in full_card, full_card[:200]
        for card in cards:
            assert _meta(card).get("is_catalogue_card") is True

        print("OVERSIZED_CATALOG", {"segments": len(cards), "full_card_chars": len(full_card)})


    # ---------------------------------------------------------------------------
    # Scenario 5 — engine switch: the card is regenerated, never duplicated
    # ---------------------------------------------------------------------------
    async def _scenario_engine_switch_regenerates_card_without_duplicating(client, agent_key, mp):
        cat = await _configure_agent(client, agent_key, EFFICIENT_ENGINE)
        response = await _ingest_sample(client, agent_key)
        assert response.status_code == 200, response.text

        first = await _card_points(cat, DECLARATIVE, SAMPLE_NAME)
        assert len(first) == 1, [(_meta(c), c.id) for c in first]
        first_ids = {c.id for c in first}

        # switch the SYSTEM ingestion engine to the core one and re-ingest
        await _set_engine(client, agent_key, CORE_ENGINE)
        spy = _install_card_hook_spy(cat.plugin_manager, mp)
        response = await _ingest_sample(client, agent_key)
        assert response.status_code == 200, response.text
        assert spy == {"splits": 1, "stores": 1, "cache_writes": 1}, spy

        second = await _card_points(cat, DECLARATIVE, SAMPLE_NAME)
        assert len(second) == 1, [(_meta(c), c.id) for c in second]

        print("SWITCH_CATALOG", {"first_ids": first_ids, "second_ids": {c.id for c in second}})


    # ---------------------------------------------------------------------------
    # Scenario 6 — images extracted from a file never leak into the prompt list
    # ---------------------------------------------------------------------------
    async def _scenario_derived_image_file_is_excluded_from_prompt_list(client, agent_key, mp):
        cat = await _configure_agent(client, agent_key, CORE_ENGINE)
        response = await _ingest_sample(client, agent_key)
        assert response.status_code == 200, response.text

        # reproduce the multimodal-ingestion artifact shape on the real stores: the
        # extracted image is SAVED as a file (so the FileManager lists it) and
        # stored as a point with image=True and image_file != source.
        derived = "sample_img_0_deadbeef.png"
        await cat.save_file(b"\x89PNG\r\n\x1a\n", "image/png", derived, None)
        vector = await asyncio.to_thread((await cat.embedder()).embed_query, "image")
        await cat.vector_memory_handler.add_points_to_tenant(
            collection_name=DECLARATIVE,
            points=[PointStruct(
                id=uuid.uuid4().hex,
                vector=vector,
                payload={
                    "page_content": f"[Image] {SAMPLE_NAME}",
                    "metadata": {
                        "source": SAMPLE_NAME, "when": time.time(),
                        "image": True, "image_file": derived,
                    },
                },
            )],
        )
        assert {p.name for p in cat.file_manager.list_files(agent_key)} >= {SAMPLE_NAME, derived}

        prompt = await _prompt_list(cat)
        assert prompt.count(SAMPLE_NAME) == 1, prompt
        assert derived not in prompt, prompt

        print("IMAGE_EXCLUDED", {"derived_image": derived, "listed_source_once": True})


    # ---------------------------------------------------------------------------
    # Scenario 6b — card id is tenant/chat scoped (B1) and survives a re-embed (B2)
    # ---------------------------------------------------------------------------
    async def _scenario_same_source_in_two_agents_gets_distinct_card_ids(client, agent_key, mp):
        """The same file name uploaded to two agents must not collide (B1, phase)."""
        agent_a = f"catalog_dual_agents_a_{uuid.uuid4().hex[:8]}"
        agent_b = f"catalog_dual_agents_b_{uuid.uuid4().hex[:8]}"

        cat_a = await _configure_agent(client, agent_a, EFFICIENT_ENGINE)
        cat_b = await _configure_agent(client, agent_b, EFFICIENT_ENGINE)

        assert (await _ingest_sample(client, agent_a)).status_code == 200
        assert (await _ingest_sample(client, agent_b)).status_code == 200

        cards_a = await _card_points(cat_a, DECLARATIVE, SAMPLE_NAME)
        cards_b = await _card_points(cat_b, DECLARATIVE, SAMPLE_NAME)
        assert len(cards_a) == 1, [(_meta(c), c.id) for c in cards_a]
        assert len(cards_b) == 1, [(_meta(c), c.id) for c in cards_b]

        ids_a = {c.id for c in cards_a}
        ids_b = {c.id for c in cards_b}
        assert ids_a.isdisjoint(ids_b), (ids_a, ids_b)

        print("TENANT_SCOPED_CATALOG", {"agent_a": sorted(ids_a), "agent_b": sorted(ids_b)})


    async def _scenario_same_source_in_two_chats_gets_distinct_card_ids(client, agent_key, mp):
        """Two chats of ONE agent get two distinct cards (B1, classic hooks)."""
        cat = await _configure_agent(client, agent_key, CORE_ENGINE)
        chat_1 = f"catalog_chat_{uuid.uuid4().hex[:8]}"
        chat_2 = f"catalog_chat_{uuid.uuid4().hex[:8]}"

        assert (await _ingest_sample(client, agent_key, chat_id=chat_1)).status_code == 200
        assert (await _ingest_sample(client, agent_key, chat_id=chat_2)).status_code == 200

        cards = await _card_points(cat, EPISODIC, SAMPLE_NAME)
        assert len(cards) == 2, [(_meta(c).get("chat_id"), c.id) for c in cards]
        by_chat = {_meta(c).get("chat_id"): c.id for c in cards}
        assert set(by_chat) == {chat_1, chat_2}, by_chat
        assert len(set(by_chat.values())) == 2, by_chat

        print("CHAT_SCOPED_CATALOG", by_chat)


    async def _scenario_reembed_keeps_card_id_and_does_not_duplicate(client, agent_key, mp):
        """A re-embed rebuild keeps the card id, so it never becomes a duplicate (B2)."""
        from cat.core_plugins.cat_efficient_ingestion import reembed

        cat = await _configure_agent(client, agent_key, EFFICIENT_ENGINE)
        assert (await _ingest_sample(client, agent_key)).status_code == 200

        cards = await _card_points(cat, DECLARATIVE, SAMPLE_NAME)
        assert len(cards) == 1, [(_meta(c), c.id) for c in cards]
        original_ids = {c.id for c in cards}

        # Emulate the re-embed rebuild the embedding phase performs: it reconstructs
        # every stored point of the source from its payload and re-stores it.
        points = await _points(cat, DECLARATIVE, {"source": SAMPLE_NAME})
        embedder = await cat.embedder()
        rebuilt = await reembed._embed_phase(
            cat, DECLARATIVE, SAMPLE_NAME, points, embedder, None, cat, rabbit_hole=None,
        )
        rebuilt_cards = [
            p for p in rebuilt
            if (p.payload or {}).get("metadata", {}).get("is_catalogue_card") is True
        ]
        assert len(rebuilt_cards) == 1, [(_meta(p), p.id) for p in rebuilt_cards]
        assert {_norm_id(p.id) for p in rebuilt_cards} == {_norm_id(i) for i in original_ids}, (
            "the re-embed rebuild lost the card id",
            original_ids,
            {p.id for p in rebuilt_cards},
        )

        # After an embedding change the catalog_cards phase is stale and re-runs: it
        # must UPSERT the same id, never add a second card.
        run_hook = next(
            h for h in cat.plugin_manager.plugins["cat_alog"].hooks
            if h.name == "ingestion_phase_run"
        )
        result = await run_hook.function("catalog_cards", SAMPLE_NAME, {}, cat)
        assert result == {"status": "done"}, result

        cards = await _card_points(cat, DECLARATIVE, SAMPLE_NAME)
        assert len(cards) == 1, [(_meta(c), c.id) for c in cards]
        assert {_norm_id(c.id) for c in cards} == {_norm_id(i) for i in original_ids}, (
            "re-embed + phase re-run changed the card id",
            original_ids,
            {c.id for c in cards},
        )

        print("REEMBED_CATALOG", {"cards": len(cards), "ids": sorted(original_ids)})


    async def _scenario_same_source_in_two_chats_phase_mode_is_isolated(client, agent_key, mp):
        """B1: the phase keeps two chats of one filename isolated (two cards)."""
        cat = await _configure_agent(client, agent_key, EFFICIENT_ENGINE)
        chat_1 = f"catalog_chat_{uuid.uuid4().hex[:8]}"
        chat_2 = f"catalog_chat_{uuid.uuid4().hex[:8]}"
        text_1 = ("alpha chat only " * 60).encode()
        text_2 = ("beta chat only " * 60).encode()

        # The phase worker resolves each chat's StrayCat by its conversation owner,
        # so the conversations must exist (the admin API key bypasses the ownership
        # claim the UI user path performs). Attach both chats to a real user.
        user = await create_new_user(
            client,
            username=f"catalog_user_{uuid.uuid4().hex[:8]}",
            headers=_headers(agent_key),
        )
        for chat in (chat_1, chat_2):
            assert await crud_conversations.claim_conversation(agent_key, chat, user["id"]) == user["id"]

        assert (
            await _ingest(client, agent_key, SAMPLE_NAME, text_1, chat_id=chat_1)
        ).status_code == 200
        assert (
            await _ingest(client, agent_key, SAMPLE_NAME, text_2, chat_id=chat_2)
        ).status_code == 200

        cards = await _card_points(cat, EPISODIC, SAMPLE_NAME)
        assert len(cards) == 2, [(_meta(c).get("chat_id"), c.id) for c in cards]
        by_chat = {_meta(c).get("chat_id"): c for c in cards}
        assert set(by_chat) == {chat_1, chat_2}, by_chat.keys()
        assert by_chat[chat_1].id != by_chat[chat_2].id, by_chat

        full_1 = _meta(by_chat[chat_1]).get("full_card") or ""
        full_2 = _meta(by_chat[chat_2]).get("full_card") or ""
        assert "alpha chat only" in full_1 and "beta chat only" not in full_1, full_1[:200]
        assert "beta chat only" in full_2 and "alpha chat only" not in full_2, full_2[:200]
        assert await _card_points(cat, DECLARATIVE, SAMPLE_NAME) == []

        print("CHAT_SCOPED_PHASE_CATALOG", {
            "segments": len(cards),
            "ids": {chat: c.id for chat, c in by_chat.items()},
        })


    async def _scenario_reembed_keeps_oversized_card_ids(client, agent_key, mp):
        """B2: a re-embed must not re-split the card, so count and ids stay stable."""
        from cat.core_plugins.cat_efficient_ingestion import reembed

        # test-only instrumentation: force the phase to segment the oversized card.
        mp.setattr(Embeddings, "max_input_tokens", 100)
        cat = await _configure_agent(client, agent_key, EFFICIENT_ENGINE)
        long_text = ("Alice was beginning to get very tired of sitting by her sister. " * 220).encode()

        assert (await _ingest(client, agent_key, LONG_NAME, long_text)).status_code == 200

        cards = await _card_points(cat, DECLARATIVE, LONG_NAME)
        assert len(cards) > 1, [(_meta(c), c.id) for c in cards]
        original_ids = sorted(c.id for c in cards)
        cat_alog = _cat_alog_module(cat)
        for card in cards:
            meta = _meta(card)
            assert meta.get("is_catalogue_card_segment") is True, meta
            assert isinstance(meta.get("start_index"), int), meta
            assert _norm_id(card.id) == _norm_id(cat_alog._catalogue_card_point_id(
                agent_key, None, LONG_NAME, start_index=meta["start_index"]
            ))

        points = await _points(cat, DECLARATIVE, {"source": LONG_NAME})
        embedder = await cat.embedder()
        # A LOWER limit after the fact: without the card-segment exemption this
        # re-split would mint a different segment count and new ids.
        mp.setattr(embedder, "max_input_tokens", 20, raising=False)
        rebuilt = await reembed._embed_phase(
            cat, DECLARATIVE, LONG_NAME, points, embedder, None, cat, rabbit_hole=None,
        )
        rebuilt_cards = [
            p for p in rebuilt
            if (p.payload or {}).get("metadata", {}).get("is_catalogue_card") is True
        ]
        assert len(rebuilt_cards) == len(cards), [p.id for p in rebuilt_cards]
        assert sorted(_norm_id(p.id) for p in rebuilt_cards) == sorted(
            _norm_id(i) for i in original_ids
        ), (
            "the re-embed re-split the card and lost its ids",
            original_ids,
            sorted(p.id for p in rebuilt_cards),
        )

        print("REEMBED_OVERSIZED_CATALOG", {
            "segments": len(cards), "ids": original_ids,
        })


    async def _scenario_phase_replaces_segments_when_embedder_window_changes(client, agent_key, mp):
        """F2: a changed embedder window REPLACES the card segments (no orphans).

        Lowering the window re-splits the card into more segments whose
        start_index-derived ids do not match the stored ones. The phase must delete
        this source's old cards in the SAME collection + chat before writing the new
        ones: the stored segments must then be EXACTLY those a fresh split of the
        same card text produces. Other sources and other chats must stay untouched.
        """
        from langchain_core.documents import Document

        from cat.core_plugins.cat_efficient_ingestion.split import split_oversized

        mp.setattr(Embeddings, "max_input_tokens", 100)
        cat = await _configure_agent(client, agent_key, EFFICIENT_ENGINE)
        cat_alog = _cat_alog_module(cat)
        run_hook = next(
            h for h in cat.plugin_manager.plugins["cat_alog"].hooks
            if h.name == "ingestion_phase_run"
        )

        long_text = ("Alice was beginning to get very tired of sitting by her sister. " * 220).encode()
        other_name = "other_catalog.txt"
        other_text = b"a different source that must never be touched"
        other_chat_name = "other_chat_catalog.txt"
        chat_1 = f"catalog_chat_{uuid.uuid4().hex[:8]}"
        chat_2 = f"catalog_chat_{uuid.uuid4().hex[:8]}"

        user = await create_new_user(
            client,
            username=f"catalog_user_{uuid.uuid4().hex[:8]}",
            headers=_headers(agent_key),
        )
        for chat in (chat_1, chat_2):
            assert await crud_conversations.claim_conversation(agent_key, chat, user["id"]) == user["id"]

        assert (await _ingest(client, agent_key, LONG_NAME, long_text)).status_code == 200
        assert (await _ingest(client, agent_key, other_name, other_text)).status_code == 200
        assert (
            await _ingest(client, agent_key, LONG_NAME, long_text, chat_id=chat_1)
        ).status_code == 200
        assert (
            await _ingest(client, agent_key, LONG_NAME, long_text, chat_id=chat_2)
        ).status_code == 200
        assert (
            await _ingest(client, agent_key, other_chat_name, other_text, chat_id=chat_2)
        ).status_code == 200

        async def _ids(collection, chat_id, source=LONG_NAME):
            return {
                p.id for p in await _card_points(cat, collection, source)
                if _meta(p).get("chat_id") == chat_id
            }

        async def _write(collection):
            return await _card_points(cat, collection, LONG_NAME)

        def _expected_offsets(card_text, embedder):
            segments = split_oversized(
                [Document(page_content=card_text, metadata={})], embedder,
            )
            normalized = " ".join(card_text.split())
            cursor = 0
            offsets = []
            for segment in segments:
                offset = normalized.find(segment.page_content, cursor)
                if offset < 0:
                    offset = cursor
                offsets.append(offset)
                cursor = offset + len(segment.page_content)
            return offsets

        async def _run_and_check(expected_more_than=None, expected_count=None):
            result = await run_hook.function("catalog_cards", LONG_NAME, {}, cat)
            assert result == {"status": "done"}, result
            agent_cards = await _write(DECLARATIVE)
            chat_cards = await _write(EPISODIC)
            assert {_meta(c).get("chat_id") for c in agent_cards} == {None}, (
                "agent-scoped cards leaked a chat scope"
            )
            assert {_meta(c).get("chat_id") for c in chat_cards} == {chat_1, chat_2}, (
                "chat-scoped cards leaked",
                {_meta(c).get("chat_id") for c in chat_cards},
            )
            full_card = _meta(agent_cards[0]).get("full_card")
            embedder = await cat.embedder()
            expected = _expected_offsets(full_card, embedder)
            if expected_count is not None:
                assert len(expected) == expected_count, (len(expected), expected_count)
            if expected_more_than is not None:
                assert len(expected) > expected_more_than, (len(expected), expected_more_than)
            # The stored agent-scoped segments are EXACTLY the fresh split: same
            # count, same start_index set, same deterministic ids -> no orphan.
            assert len(agent_cards) == len(expected), (
                "stored segment count != fresh segment count",
                len(agent_cards), len(expected),
            )
            assert sorted(_meta(c)["start_index"] for c in agent_cards) == sorted(expected)
            assert {_norm_id(c.id) for c in agent_cards} == {
                _norm_id(cat_alog._catalogue_card_point_id(
                    agent_key, None, LONG_NAME, start_index=offset,
                ))
                for offset in expected
            }
            for card in chat_cards:
                meta = _meta(card)
                assert _norm_id(card.id) == _norm_id(cat_alog._catalogue_card_point_id(
                    agent_key, meta["chat_id"], LONG_NAME, start_index=meta["start_index"],
                )), meta
                assert meta.get("is_catalogue_card_segment") is True, meta
            assert len([c for c in chat_cards if _meta(c).get("chat_id") == chat_1]) == len(expected)
            assert len([c for c in chat_cards if _meta(c).get("chat_id") == chat_2]) == len(expected)
            return agent_cards, chat_cards

        other_ids_before = await _ids(DECLARATIVE, None, source=other_name)
        assert other_ids_before, "the other source card was not produced"
        other_chat_ids_before = {
            p.id for p in await _card_points(cat, EPISODIC, other_chat_name)
        }
        assert other_chat_ids_before, "the other chat source card was not produced"

        # Baseline at window 100: an oversized, multi-segment card.
        before_dec, _ = await _run_and_check(expected_count=None)
        first_dec_ids = {c.id for c in before_dec}
        assert len(first_dec_ids) > 1, len(first_dec_ids)

        # LOWER the window -> MORE segments, the old ids must be replaced.
        mp.setattr(Embeddings, "max_input_tokens", 25)
        after_dec, _ = await _run_and_check(expected_more_than=len(first_dec_ids))
        second_dec_ids = {c.id for c in after_dec}
        assert second_dec_ids != first_dec_ids
        assert first_dec_ids - second_dec_ids, "old segment ids survived the re-run"
        print("WINDOW_LOWER_CATALOG", {
            "before": len(first_dec_ids), "after": len(second_dec_ids),
            "replaced": len(first_dec_ids - second_dec_ids),
        })

        # RAISE the window -> ONE whole card, every old segment gone.
        mp.setattr(Embeddings, "max_input_tokens", 100000)
        third_dec, third_chat = await _run_and_check(expected_count=1)
        assert len(third_dec) == 1, [(_meta(c), c.id) for c in third_dec]
        assert _norm_id(third_dec[0].id) == _norm_id(cat_alog._catalogue_card_point_id(
            agent_key, None, LONG_NAME,
        ))
        assert len(third_chat) == 2, [(_meta(c), c.id) for c in third_chat]
        print("WINDOW_RAISE_CATALOG", {"segments": len(third_dec)})

        # Other sources (agent scope and another chat) are untouched by every re-run.
        assert await _ids(DECLARATIVE, None, source=other_name) == other_ids_before
        assert {
            p.id for p in await _card_points(cat, EPISODIC, other_chat_name)
        } == other_chat_ids_before
        assert await _card_points(cat, DECLARATIVE, other_name), "other source card lost"
        assert await _card_points(cat, EPISODIC, other_chat_name), "other chat source card lost"
        assert await _card_points(cat, DECLARATIVE, LONG_NAME)

        print("PHASE_SEGMENT_REPLACE_CATALOG", {
            "lower": len(second_dec_ids), "raised": len(third_dec), "chat_segments": len(third_chat),
        })



    # ---------------------------------------------------------------------------
    # Cleanup receipt (runs last)
    # ---------------------------------------------------------------------------
    async def _cleanup_receipt():
        """No leftover symlinks, seeded keys/files/dirs or Neo4j tenant."""
        for link in list(CREATED_LINKS):
            assert not os.path.lexists(link), link
        assert not PLUGIN_LINK.exists() and not PLUGIN_LINK.is_symlink()

        if Path("tests/data").exists():
            assert not list(Path("tests/data/storage").glob(f"{SCOPE_PREFIX}*"))
        assert not list(Path("/tmp").glob(f"{SCOPE_PREFIX}*"))

        utils.singleton.instances.clear()
        _assert_test_db()
        assert await crud.scan_keys(f"agents:{SCOPE_PREFIX}*") == []
        assert await crud.scan_keys("system:*") == []

        await _assert_neo4j_reachable()
        assert await _count_tenant_prefix(SCOPE_PREFIX) == 0

        print("CLEANUP_RECEIPT", {"redis_catalog_keys": 0, "neo4j_catalog_tenants": 0, "seeded_links": 0})


    SCENARIOS = [
        ("classic_mode_one_card_ordered_next_and_prompt_once",
         _scenario_classic_mode_one_card_ordered_next_and_prompt_once),
        ("efficient_mode_phase_builds_one_card_and_never_fires_classic_hooks",
         _scenario_efficient_mode_phase_builds_one_card_and_never_fires_classic_hooks),
        ("episodic_card_is_chat_scoped", _scenario_episodic_card_is_chat_scoped),
        ("oversized_card_is_segmented_with_full_card", _scenario_oversized_card_is_segmented_with_full_card),
        ("engine_switch_regenerates_card_without_duplicating",
         _scenario_engine_switch_regenerates_card_without_duplicating),
        ("derived_image_file_is_excluded_from_prompt_list",
         _scenario_derived_image_file_is_excluded_from_prompt_list),
        ("same_source_in_two_agents_gets_distinct_card_ids",
         _scenario_same_source_in_two_agents_gets_distinct_card_ids),
        ("same_source_in_two_chats_gets_distinct_card_ids",
         _scenario_same_source_in_two_chats_gets_distinct_card_ids),
        ("reembed_keeps_card_id_and_does_not_duplicate",
         _scenario_reembed_keeps_card_id_and_does_not_duplicate),
        ("same_source_in_two_chats_phase_mode_is_isolated",
         _scenario_same_source_in_two_chats_phase_mode_is_isolated),
        ("reembed_keeps_oversized_card_ids", _scenario_reembed_keeps_oversized_card_ids),
        ("phase_replaces_segments_when_embedder_window_changes",
         _scenario_phase_replaces_segments_when_embedder_window_changes),
    ]


    async def _main():
        total = len(SCENARIOS) + 1
        passed = 0
        for name, func in SCENARIOS:
            try:
                await _run_scenario(name, func)
                passed += 1
                print(f"PASS {name}")
            except Exception:  # noqa: BLE001 - report every scenario, never hide
                traceback.print_exc()
                print(f"FAIL {name}")
        try:
            await _cleanup_receipt()
            passed += 1
            print("PASS cleanup_receipt")
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            print("FAIL cleanup_receipt")
        print(f"\ntest_dual_mode: {passed}/{total} passed")
        if passed != total:
            raise SystemExit(1)


    try:
        asyncio.run(_main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"FAIL test_dual_mode: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
