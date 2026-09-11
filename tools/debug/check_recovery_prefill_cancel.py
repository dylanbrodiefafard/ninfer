"""Disconnect a captured loop during its recovery prefill, with a live peer.

Requires an isolated diagnostic server with 500 ms throughput logs.
No returned tool is executed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from pathlib import Path
import socket
import threading
import time
import urllib.request

from jsonschema import validate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture', type=Path, required=True)
    ap.add_argument('--probe', type=Path, required=True)
    ap.add_argument('--server-log', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--port', type=int, default=19002)
    ap.add_argument('--temperature', type=float, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tokens', type=int, default=32000)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    body = json.loads(args.fixture.read_text())
    body.pop('max_tokens', None)
    body.update(model='loop-numerics', max_completion_tokens=args.tokens, stream=True,
                enable_thinking=True, temperature=args.temperature, seed=args.seed)
    payload = '\n'.join(f'peer item {i}' for i in range(2000))
    peer_schema = dict(type='object', properties={'payload': {'const': payload}},
                       required=['payload'], additionalProperties=False)
    peer = dict(model='loop-numerics', max_completion_tokens=16384, seed=42, temperature=2,
                enable_thinking=False, stream=False,
                messages=[dict(role='user', content='Call peer with exactly its declared payload.')],
                tools=[dict(type='function', function=dict(name='peer', parameters=peer_schema))])
    ready, cancelled = threading.Event(), threading.Event()
    connection = []
    events = []

    def request(value):
        return urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/chat/completions',
            data=json.dumps(value).encode(), headers={'Content-Type': 'application/json'})

    def read_stream():
        try:
            with urllib.request.urlopen(request(body), timeout=240) as response:
                connection.append(response)
                ready.set()
                for line in response:
                    if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                        continue
                    event = json.loads(line[6:])
                    events.append(event)
                    assert 'error' not in event, event
                    assert not any(c.get('delta', {}).get('tool_calls') for c in event.get('choices', [])), (
                        'repeated call was published before recovery cancellation', event)
        except (OSError, http.client.IncompleteRead):
            if not cancelled.is_set():
                raise
        finally:
            ready.set()

    def complete(label, value):
        with urllib.request.urlopen(request(value), timeout=240) as response:
            result = json.load(response)
        (args.output / (label + '.json')).write_text(json.dumps(result, indent=2) + '\n')
        choice = result['choices'][0]
        assert choice['finish_reason'] == 'tool_calls', choice
        schemas = {t['function']['name']: t['function']['parameters'] for t in value['tools']}
        for call in choice['message']['tool_calls']:
            f = call['function']
            validate(json.loads(f['arguments']), schemas[f['name']])
        return result

    with ThreadPoolExecutor(max_workers=2) as pool, args.server_log.open() as log:
        log.seek(0, 2)
        stream = pool.submit(read_stream)
        if not ready.wait(timeout=120) or not connection:
            stream.result()
            raise AssertionError('stream did not become available')
        # Submit the peer after the captured request starts. Its long constrained
        # call keeps it decode-ready through the captured request's first attempt.
        survivor = pool.submit(complete, 'peer', peer)
        seen_decode = False
        observations = []
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            line = log.readline()
            if not line:
                if stream.done():
                    stream.result()
                    raise AssertionError('stream completed before a recovery prefill was observed')
                time.sleep(0.05)
                continue
            if 'throughput interval=' not in line:
                continue
            observations.append(line.rstrip())
            if 'prefilling=0' in line and ('decode_ready=1' in line or 'decode_ready=2' in line):
                seen_decode = True
            if seen_decode and 'prefilling=1' in line:
                # The peer's initial prefill is also a phase transition. Once
                # both requests have entered a packed round, only an internal
                # retry can introduce another prefill in this two-request check.
                if not any('avg_decode_batch=2.00' in prior for prior in observations):
                    continue
                if 'running=2' not in line or 'decode_ready=1' not in line:
                    raise AssertionError('peer was not active during the observed retry prefill')
                cancelled.set()
                connection[0].fp.raw._sock.shutdown(socket.SHUT_RDWR)
                break
        else:
            cancelled.set()
            connection[0].fp.raw._sock.shutdown(socket.SHUT_RDWR)
            raise AssertionError('no recovery prefill observed within the diagnostic deadline')
        stream.result(timeout=30)
        survivor.result(timeout=180)
    (args.output / 'stream.json').write_text(json.dumps(events, indent=2) + '\n')
    (args.output / 'prefill-observation.json').write_text(json.dumps(observations, indent=2) + '\n')
    probe = json.loads(args.probe.read_text())
    probe.update(model='loop-numerics', max_completion_tokens=16384, seed=42, temperature=2,
                 enable_thinking=True, stream=False)
    complete('subsequent', probe)
    print('disconnected during observed recovery prefill; peer and subsequent tool requests valid', flush=True)


if __name__ == '__main__':
    main()
