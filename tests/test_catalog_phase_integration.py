"""Regression: the catalogue card must be produced against the REAL host cat.

The host removed ``cat.mad_hatter`` and exposes ``cat.plugin_manager`` instead.
CATALOG used to call ``cat.mad_hatter.get_plugin().load_settings()``; the
resulting ``AttributeError`` was swallowed by ``_run_hooks``, so the summary was
never cached, no card was created and nothing was logged as an error.

This script boots the real host app/plugin manager (no ``mad_hatter`` stub, no
plugin-manager stub). Only the LLM call is stubbed: the summary agentic workflow
returns a fixed summary.

Standalone script (plugin rules): the host imports every ``.py`` under the
plugin folder, so the ENTIRE body lives inside ``if __name__ == "__main__":``
and importing this file executes zero code. Run it as a plain script from the
host checkout:

    cd /root/git/grinning-cat-core
    .venv/bin/python /root/git/plugins/cat_alog/tests/test_catalog_phase_integration.py

Why no conftest symlink
-----------------------
The old host copy relied on ``tests/catalog/conftest.py`` to symlink
``cat_alog`` under ``cat/core_plugins/``. Here there is no pytest conftest: the
plugin is made importable under the exact dotted name the host loader derives
(``cat.core_plugins.cat_alog.*``) with an explicit ``sys.modules`` package whose
``__path__`` points at this repository, and it is then loaded/activated through
the REAL host ``Plugin``/``MadHatter`` classes. Nothing is copied or symlinked
into ``cat/``.
"""

if __name__ == "__main__":
    import asyncio
    import os
    import subprocess
    import sys
    import traceback
    import types
    import uuid
    from contextlib import ExitStack
    from inspect import iscoroutinefunction
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch

    # --- environment BEFORE importing cat.startup (it builds an app at import) ---
    # Dedicated test Redis DB only; never the production db=0 (AGENTS.md safety).
    TEST_REDIS_DB = "1"


    def _resolve_redis_host():
        """Host venv needs a RedisJSON-capable server (the host conftest expects
        CAT_REDIS_HOST to be provided by the runner). When running on this
        workspace's host, discover the dev compose redis-stack container."""
        explicit = os.environ.get("CAT_REDIS_HOST")
        if explicit:
            return explicit
        for name in ("grinning_cat_redis", "dev-grinning-cat-redis-1", "redis"):
            try:
                out = subprocess.run(
                    [
                        "docker", "inspect", "-f",
                        "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                        name,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
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

    from langchain_core.documents import Document
    from qdrant_client import AsyncQdrantClient
    from asgi_lifespan import LifespanManager

    from cat import utils
    from cat.auth import auth_utils
    import cat.looking_glass.mad_hatter.plugin as plugin_mod
    from cat.looking_glass.mad_hatter.plugin import Plugin
    from cat.mixins import ContextMixin
    from cat.routes import routes_utils
    from cat.services.factory.vector_db import QdrantHandler
    from cat.startup import create_app

    PLUGIN_ID = "cat_alog"
    #: The plugin root: this file lives in ``tests/`` after the move.
    REPO_ROOT = Path(__file__).resolve().parent.parent
    PLUGIN_DIR = REPO_ROOT
    AGENT_ID = f"catalog_phase_integration_{uuid.uuid4().hex[:10]}"


    class _FakeSummaryWorkflow:
        async def run(self, task=None, llm=None):
            return SimpleNamespace(output="FAKE SUMMARY")


    class _ExternalPlugin(Plugin):
        """Real host ``Plugin`` whose dotted module name maps this repo under
        ``cat.core_plugins.<id>`` (the folder is not physically under ``cat/``)."""

        def _get_py_filename_dotted_notation(self, py_file):
            resolved = Path(py_file).resolve()
            try:
                rel = resolved.relative_to(PLUGIN_DIR.resolve())
            except ValueError:
                return super()._get_py_filename_dotted_notation(py_file)
            return f"cat.core_plugins.{PLUGIN_ID}." + str(rel.with_suffix("")).replace(os.sep, ".")


    def _register_external_package():
        pkg_name = f"cat.core_plugins.{PLUGIN_ID}"
        if pkg_name in sys.modules:
            return
        import cat.core_plugins  # noqa: F401 - ensure the parent package exists

        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(PLUGIN_DIR)]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg


    async def _load_external_plugin(agent_key):
        """Load + activate this repository through the real host ``Plugin``."""
        _register_external_package()
        original = plugin_mod.get_core_plugins_path
        # The loader's "is this a core plugin?" guard must be true for the
        # out-of-tree package so it does not purge the module we registered.
        plugin_mod.get_core_plugins_path = lambda: str(PLUGIN_DIR.parent) + "/"
        try:
            plugin = _ExternalPlugin(str(PLUGIN_DIR))
            await plugin.activate(agent_key)
        finally:
            plugin_mod.get_core_plugins_path = original
        return plugin


    def _hook_function(plugin_manager, name):
        plugin = plugin_manager.plugins[PLUGIN_ID]
        hook = next(h for h in plugin.hooks if h.name == name)
        return hook.function


    def _safe_flush_test_db():
        from cat.db.database import get_sync_db

        db = get_sync_db()
        bound_db = db.connection_pool.connection_kwargs.get("db", os.environ.get("CAT_REDIS_DB"))
        if str(bound_db) != TEST_REDIS_DB:
            raise RuntimeError(
                f"Refusing to flushdb(): bound to Redis db={bound_db!r}, expected test db={TEST_REDIS_DB!r}."
            )
        db.flushdb()


    async def _run_assertions(cheshire_cat):
        # Target host contract: `cat.mad_hatter` does not exist. This checkout
        # still carries a compatibility property, so remove it to make a
        # regression to `cat.mad_hatter` fail here exactly as it does on the host.
        mad_hatter_prop = ContextMixin.__dict__.get("mad_hatter")
        if mad_hatter_prop is not None:
            delattr(ContextMixin, "mad_hatter")
        try:
            assert not hasattr(cheshire_cat, "mad_hatter")

            plugin_manager = cheshire_cat.plugin_manager

            # The plugin must have been resolved through the REAL host plugin manager.
            assert type(plugin_manager).__name__ == "MadHatter"
            assert PLUGIN_ID in plugin_manager.plugins

            register_hook = _hook_function(plugin_manager, "rabbithole_ingestion_phases")
            cat_alog = sys.modules[register_hook.__module__]

            # Only the LLM is stubbed: settings, the vector store and the
            # registration hook are the REAL host ones.
            cheshire_cat.agentic_workflow = _FakeSummaryWorkflow()

            # (1) The core hook contributes the phase CLASS from this plugin.
            registered = register_hook([], cheshire_cat)
            assert registered == [cat_alog.CatalogCardsPhase], (
                "the catalog phase was not registered through the real plugin manager: "
                f"{registered}"
            )
            assert issubclass(registered[0], cat_alog.IngestionPhase)

            phase = registered[0]()

            # (2) settings() resolves the plugin settings through the real manager
            # (the `cat.mad_hatter` regression used to break this path).
            settings = await phase.settings(cheshire_cat)
            assert settings["max_document_chars"] == 8000
            assert settings["max_summary_words"] == 200

            source = cat_alog.IngestionSource(
                agent_id=cheshire_cat.agent_key, chat_id=None, name="real.txt",
            )
            applies, reason = await phase.applies_to(source, cheshire_cat)
            assert applies is True and reason is None

            # (3) A real chunk in the real vector memory -> a real card.
            from cat.services.memory.models import PointStruct, VectorMemoryType

            collection = str(VectorMemoryType.DECLARATIVE)
            handler = cheshire_cat.vector_memory_handler
            # The agent already created its collections with the configured
            # embedder: size the fake vectors to the real collection.
            info = await handler._client.get_collection(collection)
            vector = [0.01] * info.config.params.vectors.size
            await handler.add_points_to_tenant(collection, points=[PointStruct(
                id=uuid.uuid4().hex,
                payload={
                    "page_content": "hello world content",
                    "metadata": {"source": "real.txt", "chunk_index": 0},
                },
                vector=vector,
            )])

            class _Embedder:
                def _estimate_tokens(self, text):
                    return 1

                def embed_documents(self, texts):
                    return [vector for _ in texts]

            async def _embedder(cat):
                return _Embedder()

            original_embedder = cat_alog._resolve_embedder
            cat_alog._resolve_embedder = _embedder
            try:
                result = await phase.process(source, cheshire_cat)
            finally:
                cat_alog._resolve_embedder = original_embedder

            assert result.status == "done", f"phase did not complete: {result}"
            records, _ = await handler.get_all_tenant_points(
                collection,
                metadata={"source": "real.txt", "is_catalogue_card": True},
                with_vectors=False,
            )
            assert len(records) == 1, (
                "no catalogue card produced through the real plugin manager: "
                f"{records}"
            )
            payload = records[0].payload or {}
            metadata = payload.get("metadata") or {}
            assert "FAKE SUMMARY" in metadata.get("full_card", "")
            assert "real.txt" in metadata.get("full_card", "")

            # (4) delete() purges ONLY the source's card, never its chunk.
            await phase.delete(source, cheshire_cat)
            cards_after, _ = await handler.get_all_tenant_points(
                collection,
                metadata={"source": "real.txt", "is_catalogue_card": True},
                with_vectors=False,
            )
            chunks_after, _ = await handler.get_all_tenant_points(
                collection, metadata={"source": "real.txt"}, with_vectors=False,
            )
            assert cards_after == [], "delete must remove the source's cards"
            assert len(chunks_after) == 1, "delete must not remove the source's chunk"
        finally:
            if mad_hatter_prop is not None:
                setattr(ContextMixin, "mad_hatter", mad_hatter_prop)


    async def _main():
        memory_client = AsyncQdrantClient(":memory:")
        stack = ExitStack()
        try:
            # --- host test doubles (same contract as the old conftest) ---
            def _mock_qdrant_init(self, *args, **kwargs):
                super(QdrantHandler, self).__init__()
                self._client = memory_client
                self.save_memory_snapshots = kwargs.get("save_memory_snapshots", False)

            stack.enter_context(patch.object(QdrantHandler, "__init__", _mock_qdrant_init))
            stack.enter_context(patch.object(QdrantHandler, "close", lambda self: None))
            stack.enter_context(
                patch.object(utils, "get_plugins_path", lambda: "tests/mocks/mock_plugin_folder/")
            )
            stack.enter_context(
                patch.object(utils, "get_file_manager_root_storage_path", lambda: "tests/data/storage")
            )

            async def _noop_install_requirements(self):
                return None

            stack.enter_context(
                patch.object(Plugin, "_install_requirements", _noop_install_requirements)
            )
            stack.enter_context(
                patch.object(auth_utils, "extract_agent_id_from_request", lambda *a, **k: AGENT_ID)
            )

            async def _immediate_task(background_tasks, func, *args, **kwargs):
                if iscoroutinefunction(func):
                    await func(*args, **kwargs)
                else:
                    func(*args, **kwargs)

            stack.enter_context(
                patch.object(routes_utils, "run_background_task", _immediate_task)
            )

            utils.singleton.instances.clear()
            _safe_flush_test_db()

            app = create_app()
            async with LifespanManager(app):
                lizard = app.state.lizard
                cheshire_cat = await lizard.create_cheshire_cat(AGENT_ID)

                plugin = await _load_external_plugin(AGENT_ID)
                cheshire_cat.plugin_manager.plugins[PLUGIN_ID] = plugin

                # `get_plugin()` resolves the caller by its file location: point
                # the plugin root at this repository for the hook execution.
                with patch.object(utils, "get_plugins_path", lambda: str(PLUGIN_DIR.parent) + "/"):
                    await _run_assertions(cheshire_cat)
        finally:
            stack.close()
            utils.singleton.instances.clear()
            try:
                _safe_flush_test_db()
            except Exception:  # noqa: BLE001 - never hide the real outcome
                pass
            collections = await memory_client.get_collections()
            for collection in collections.collections:
                await memory_client.delete_collection(collection.name)
            await memory_client.close()


    try:
        asyncio.run(_main())
    except Exception as exc:  # noqa: BLE001 - report and fail
        traceback.print_exc()
        print(f"FAIL test_catalog_phase_integration: {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    print("PASS test_catalog_phase_integration")
    print("\ntest_catalog_phase_integration: 1/1 passed")
