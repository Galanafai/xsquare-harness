"""The relay: a WebSocket server the robot client connects to.

    robot client --(public_v2)--> [this relay] --(public_v2)--> Wall-X
                 <--(actions)----              <--(actions)----

For every observation the robot sends, the relay:
  1. decodes and checks it,
  2. asks the supervisor what to do (ACT / HOLD / PASS),
  3. ACT or PASS: forwards to Wall-X, checks the returned chunk,
     HOLD: answers itself with a "stay still" chunk,
  4. sends exactly one reply back, and records everything.

Rules this file enforces, each one a failure mode of the example harness:

* NEVER close the robot connection because the agent or Wall-X failed.
  Closing made the client reconnect, which restarted the task mid-motion.
  If Wall-X is down or slow we reply HOLD, built from the robot's own
  measured state, so the robot stays still and keeps asking.
* After a Wall-X timeout, throw that connection away. A late reply on a
  reused connection would be paired with the NEXT request (wrong actions).
* Never cache or replay actions.
* Forward Wall-X replies byte-for-byte unless we deliberately change them
  (HOLD, freeze idle arm), so dtypes and extra fields stay exactly as sent.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed, WebSocketException

from .action_filter import ActionFilter
from .config import HarnessConfig, load_headers
from .protocol import (
    ARMS,
    MAX_FRAME_BYTES,
    ProtocolError,
    build_response,
    error_code,
    hold_actions,
    is_error_response,
    build_prewarm_request,
    pack,
    read_state,
    unpack,
    validate_request,
)
from .recorder import SessionRecorder
from .supervisor import ACT, HOLD, PASS, Decision, Supervisor


class UpstreamError(RuntimeError):
    """Wall-X could not produce a reply for this request."""


class WallXUpstream:
    """One connection to Wall-X, rebuilt whenever it becomes untrustworthy."""

    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg.wallx
        self.headers = load_headers(cfg.wallx)
        self.ws = None
        self.metadata: dict | None = None
        self.metadata_raw: bytes | None = None
        self.requests_on_this_connection = 0
        self.cache_path = Path(cfg.recorder.dir) / "wallx_metadata.msgpack"

    async def connect(self) -> None:
        await self.close()
        try:
            self.ws = await connect(
                self.cfg.endpoint,
                additional_headers=self.headers,
                proxy=None,
                ping_interval=None,  # sync inference can take long; don't let pings kill it
                compression=None,
                open_timeout=self.cfg.open_timeout_s,
                close_timeout=2,
                max_size=MAX_FRAME_BYTES,
            )
            raw = await asyncio.wait_for(self.ws.recv(), self.cfg.open_timeout_s)
            meta = unpack(raw)
            if not isinstance(meta, dict) or "error" in meta:
                raise UpstreamError(f"bad handshake: {str(meta)[:200]}")
        except (OSError, asyncio.TimeoutError, WebSocketException, ProtocolError) as err:
            await self.close()
            raise UpstreamError(f"connect failed: {type(err).__name__}: {err}") from err
        self.metadata, self.metadata_raw = meta, raw
        self.requests_on_this_connection = 0
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(raw)

    def cached_metadata_raw(self) -> bytes:
        """Handshake to give the robot when Wall-X is unreachable right now."""
        if self.metadata_raw:
            return self.metadata_raw
        if self.cache_path.exists():
            return self.cache_path.read_bytes()
        # UNVERIFIED that the client accepts this; it only reaches the robot
        # if Wall-X has never been reachable from this laptop.
        return pack({"model": "wall-x", "note": "xharness: upstream unavailable"})

    async def infer(self, frame: bytes) -> bytes:
        """Send one request, return the raw reply frame. Raises UpstreamError."""
        attempts = 1 + max(0, self.cfg.reconnect_attempts)
        last: Exception | None = None
        for _ in range(attempts):
            try:
                if self.ws is None:
                    await self.connect()
                timeout = (
                    self.cfg.first_request_timeout_s
                    if self.requests_on_this_connection == 0
                    else self.cfg.request_timeout_s
                )
                await asyncio.wait_for(self.ws.send(frame), timeout)
                reply = await asyncio.wait_for(self.ws.recv(), timeout)
                self.requests_on_this_connection += 1
                if isinstance(reply, str):
                    # Server crash: text traceback, then close 1011.
                    raise UpstreamError(f"server sent text frame: {reply[:200]!r}")
                return reply
            except (asyncio.TimeoutError, ConnectionClosed, OSError, UpstreamError, WebSocketException) as err:
                last = err
                await self.close()  # never reuse a connection after a failure
        raise UpstreamError(f"{type(last).__name__}: {last}")

    async def close(self) -> None:
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:  # noqa: BLE001
                pass
        self.ws = None


class Relay:
    def __init__(self, cfg: HarnessConfig, supervisor: Supervisor, verbose: bool = True, prewarm: bool = True):
        self.cfg = cfg
        self.supervisor = supervisor
        self.filter = ActionFilter(cfg.filter)
        self.verbose = verbose
        self.prewarm_enabled = prewarm
        self.prewarm_result: dict | None = None
        self.sessions = 0

    async def serve_forever(self) -> None:
        async with serve(
            self.handle,
            self.cfg.listen.host,
            self.cfg.listen.port,
            ping_interval=None,
            compression=None,
            max_size=MAX_FRAME_BYTES,
        ):
            self._say(f"listening on ws://{self.cfg.listen.host}:{self.cfg.listen.port}")
            warm = asyncio.create_task(self.prewarm()) if self.prewarm_enabled else None
            try:
                await asyncio.Future()
            finally:
                if warm is not None:
                    warm.cancel()

    async def prewarm(self) -> dict:
        """Send one synthetic request to Wall-X so a cold start never lands on a real one.

        Measured: the first call after idle took 4.0 s, later calls 0.6 s, and
        the warm-up belongs to the server, not the connection. It runs on its
        OWN connection and its reply is dropped here: nothing from it can reach
        the robot. Any failure is logged and ignored.
        """
        upstream = WallXUpstream(self.cfg)
        recorder = SessionRecorder(self.cfg.recorder, label="prewarm")
        started = time.monotonic()
        result: dict = {"ok": False}
        try:
            raw = await upstream.infer(pack(build_prewarm_request()))
            reply = unpack(raw)
            result = {"ok": not is_error_response(reply), "infer_ms": (reply.get("server_timing") or {}).get("infer_ms")}
            if not result["ok"]:
                result["error"] = str(reply.get("error"))[:200]
        except (UpstreamError, ProtocolError) as err:
            result["error"] = str(err)[:200]
        except Exception as err:  # noqa: BLE001 - pre-warm must never take the relay down
            result["error"] = f"{type(err).__name__}: {err}"[:200]
        finally:
            await upstream.close()
        result["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
        recorder.event("prewarm", **result)
        recorder.close()
        self.prewarm_result = result
        status = "ok" if result["ok"] else f"FAILED ({result.get('error')})"
        self._say(f"pre-warm {status}: {result['latency_ms']:.0f} ms (reply discarded)")
        return result

    # ------------------------------------------------------------------

    async def handle(self, downstream: ServerConnection) -> None:
        self.sessions += 1
        upstream = WallXUpstream(self.cfg)
        recorder = SessionRecorder(self.cfg.recorder, label=f"session{self.sessions}")
        recorder.event("session_start", endpoint=self.cfg.wallx.endpoint, filter_mode=self.cfg.filter.mode)
        try:
            try:
                await upstream.connect()
                meta_raw = upstream.metadata_raw
                recorder.event("handshake", metadata=upstream.metadata)
            except UpstreamError as err:
                meta_raw = upstream.cached_metadata_raw()
                recorder.event("handshake_failed", error=str(err))
                self._say(f"WARNING Wall-X unreachable at session start ({err}); robot will get HOLD")
            await downstream.send(meta_raw)

            index = 0
            async for frame in downstream:
                index += 1
                reply = await self._one_request(index, frame, upstream, recorder)
                await downstream.send(reply)
        except ConnectionClosed:
            recorder.event("session_closed_by_client")
            self._say("robot client disconnected (progress is kept; stop the run in the WebUI if needed)")
        finally:
            await upstream.close()
            recorder.event("session_end")
            recorder.close()

    async def _one_request(self, index: int, frame, upstream: WallXUpstream, rec: SessionRecorder) -> bytes:
        started = time.monotonic()

        # 1) Decode and read state. Without state we cannot even build HOLD.
        try:
            request = unpack(frame)
            state = read_state(request)
            embodiment = request["embodiment"]
            goal = request["instruction"]
        except (ProtocolError, KeyError, TypeError) as err:
            rec.event("bad_request", index=index, error=str(err))
            self._say(f"[{index:04d}] bad request from client: {err}")
            # Let Wall-X judge it: it returns a documented error the client handles.
            try:
                return await upstream.infer(frame)
            except UpstreamError:
                return pack({"error": {"code": "harness_bad_request", "message": str(err)[:200]}})
        try:
            validate_request(request)
        except ProtocolError as err:
            rec.event("request_warning", index=index, error=str(err))
        rec.frames(index, request)
        rec.event("request", index=index, goal=goal, state={a: vars(state[a]) for a in ARMS})

        # 2) Ask the supervisor. It must never take the session down.
        try:
            decision = await self.supervisor.decide(request, state)
        except Exception as err:  # noqa: BLE001
            decision = Decision(HOLD, reason=f"supervisor error: {type(err).__name__}: {err}")
        for ev in decision.events:
            rec.event("supervisor", index=index, **ev)

        # 3) Produce exactly one reply.
        if decision.kind == HOLD:
            reply = pack(build_response(embodiment, hold_actions(state)))
            outcome = "hold"
        else:
            if decision.kind == ACT:
                request["instruction"] = decision.instruction
                out_frame = pack(request)
            else:  # PASS: byte-exact forward of the client's request
                out_frame = frame
            reply, outcome = await self._forward(index, out_frame, request, state, embodiment, goal, decision, upstream, rec)

        elapsed = round((time.monotonic() - started) * 1000, 1)
        rec.event(
            "reply",
            index=index,
            outcome=outcome,
            decision=decision.kind,
            stage=decision.stage_id,
            instruction=decision.instruction,
            reason=decision.reason,
            elapsed_ms=elapsed,
        )
        self._status(index, decision, outcome, elapsed)
        return reply

    async def _forward(self, index, out_frame, request, state, embodiment, goal, decision, upstream, rec):
        try:
            raw = await upstream.infer(out_frame)
            response = unpack(raw)
        except (UpstreamError, ProtocolError) as err:
            rec.event("upstream_failure", index=index, error=str(err))
            if self.cfg.hold.on_upstream_failure:
                return pack(build_response(embodiment, hold_actions(state))), "hold_upstream_failure"
            return pack({"error": {"code": "upstream_unavailable", "message": str(err)[:200]}}), "error_upstream"

        if is_error_response(response):
            # Documented error: forward unchanged, the client knows these.
            rec.event("wallx_error", index=index, code=error_code(response), detail=str(response.get("error"))[:300])
            return raw, f"wallx_error:{error_code(response)}"

        result = self.filter.check(response, embodiment, state, decision.active_arm)
        rec.event("chunk", index=index, ok=result.ok, violations=result.violations, stats=result.stats,
                  timing=response.get("server_timing"))
        if not result.ok and (self.cfg.filter.mode == "enforce" or any(v.startswith("protocol") for v in result.violations)):
            # Malformed chunks are never forwarded; other violations only in enforce mode.
            return pack(build_response(embodiment, hold_actions(state))), "hold_filter"

        try:
            actions = response["actions"]
            self.supervisor.note_commanded_gripper(goal, {a: float(actions[a]["gripper"][-1]) for a in ARMS})
        except (KeyError, TypeError, IndexError):
            pass

        if result.modified_actions is not None:
            return pack(build_response(embodiment, result.modified_actions)), "forwarded_modified"
        return raw, "forwarded" if result.ok else "forwarded_with_violations"

    # ------------------------------------------------------------------ console

    def _status(self, index: int, decision: Decision, outcome: str, elapsed_ms: float) -> None:
        verify = next((e for e in decision.events if e.get("event") == "verify"), None)
        v = f" verify={verify['answer']}" if verify else ""
        text = f" '{decision.instruction[:50]}'" if decision.kind != HOLD else ""
        self._say(f"[{index:04d}] {decision.kind.upper():4s} {decision.stage_id or '-':16s} {outcome:24s} "
                  f"{elapsed_ms:7.0f} ms{v}{text}")

    def _say(self, text: str) -> None:
        if self.verbose:
            print(text, flush=True)


__all__ = ["Relay", "WallXUpstream", "UpstreamError", "ACT", "HOLD", "PASS"]
