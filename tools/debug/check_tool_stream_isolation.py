"""Real Engine tool-mask isolation and disconnect check; never executes tools."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.request

from jsonschema import validate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--server', type=Path, default=Path('/build/apps/ninfer-serve'))
    ap.add_argument('--spec', choices=('off', 'mtp', 'dflash'), default='dflash')
    ap.add_argument('--no-cuda-graph', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    weights = args.weights.resolve(strict=True)
    server = args.server.resolve(strict=True)
    output = args.output.resolve()
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', 18002)) == 0:
            raise RuntimeError('diagnostic port 18002 already has a listener')
    output.mkdir(parents=True, exist_ok=False)
    command = [str(server), str(weights), '--host', '127.0.0.1', '--port', '18002',
               '--model-id', 'tool-isolation', '--max-context', '16384',
               '--max-concurrency', '2', '--kv-dtype', 'nvfp4', '--kv-capacity', 'auto',
               '--kv-disk-capacity', 'off', '--kv-ram-capacity', 'off', '--no-prefix-reuse',
               '--temperature', '2', '--log-stats-interval-ms', '500',
               '--request-log-jsonl', str(output / 'requests.jsonl')]
    if args.spec != 'off':
        command += ['--spec', args.spec, '--draft-tokens', '5']
        if args.spec == 'dflash':
            command += ['--lm-head-draft']
    if args.no_cuda_graph:
        command += ['--no-cuda-graph']

    def fixture(name, count=700):
        payload = '  ' + '\n'.join(f'{name} item {i}' for i in range(count)) + '\n\n'
        schema = dict(type='object', properties={'payload': {'type': 'string', 'const': payload}},
                      required=['payload'], additionalProperties=False)
        return dict(model='tool-isolation', seed=42, temperature=2, enable_thinking=False,
                    max_completion_tokens=8192, messages=[dict(role='user', content=
                    f'Call {name} exactly once, using the exact payload declared in its schema.')],
                    tools=[dict(type='function', function=dict(name=name, parameters=schema))])

    def request(body):
        return urllib.request.Request('http://127.0.0.1:18002/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})

    def complete(label, body, barrier=None, stream=False):
        if barrier is not None:
            barrier.wait(timeout=30)
        with urllib.request.urlopen(request(dict(body, stream=stream)), timeout=180) as response:
            if not stream:
                result = json.load(response)
            else:
                events, assembled, content, usage, finish = [], {}, '', None, None
                for line in response:
                    if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                        continue
                    event = json.loads(line[6:])
                    events.append(event)
                    assert 'error' not in event, event
                    if event.get('usage'):
                        usage = event['usage']
                    for choice in event.get('choices', []):
                        finish = choice.get('finish_reason') or finish
                        delta = choice.get('delta', {})
                        content += delta.get('content') or ''
                        for call in delta.get('tool_calls', []):
                            item = assembled.setdefault(call['index'], dict(function=dict(name='', arguments='')))
                            for key in ('name', 'arguments'):
                                item['function'][key] += call.get('function', {}).get(key, '')
                (output / (label + '.events.json')).write_text(json.dumps(events, indent=2) + '\n')
                result = dict(usage=usage, choices=[dict(finish_reason=finish, message=dict(
                    content=content, tool_calls=[assembled[i] for i in sorted(assembled)]))])
        (output / (label + '.json')).write_text(json.dumps(result, indent=2) + '\n')
        choice = result['choices'][0]
        assert choice['finish_reason'] == 'tool_calls', choice
        message = choice['message']
        assert '<tool_call>' not in (message.get('content') or ''), message
        calls = message['tool_calls']
        assert len(calls) == 1, calls
        declared = body['tools'][0]['function']
        assert calls[0]['function']['name'] == declared['name'], calls
        decoded = json.loads(calls[0]['function']['arguments'])
        validate(decoded, declared['parameters'])
        assert decoded['payload'] == declared['parameters']['properties']['payload']['const']
        return result['usage']

    def disconnect(barrier):
        body = fixture('cancelled_call')
        body['enable_thinking'] = True
        body['messages'][0]['content'] = (
            'Work through a detailed comparison of CUDA attention reduction algorithms, '
            'including numerical error and synchronization, before calling cancelled_call.')
        body['stream'] = True
        barrier.wait(timeout=30)
        with urllib.request.urlopen(request(body), timeout=180) as response:
            for line in response:
                if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                    continue
                event = json.loads(line[6:])
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    assert not delta.get('tool_calls'), 'call published before cancellation point'
                    if delta.get('reasoning_content') or delta.get('content'):
                        (output / 'disconnect.json').write_text(json.dumps(event, indent=2) + '\n')
                        return 'closed after first generated text delta'
        raise AssertionError('stream ended before the cancellation point')

    (output / 'run.json').write_text(json.dumps(dict(command=command), indent=2) + '\n')
    with (output / 'server.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            for _ in range(180):
                if process.poll() is not None:
                    raise RuntimeError('server exited; inspect server.log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18002/health', timeout=1):
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError('server readiness timeout')
            with ThreadPoolExecutor(max_workers=2) as pool:
                barrier = threading.Barrier(2)
                left = pool.submit(complete, 'alpha', fixture('alpha'), barrier)
                right = pool.submit(complete, 'beta', fixture('beta'), barrier)
                results = dict(alpha=left.result(), beta=right.result())
                barrier = threading.Barrier(2)
                cancelled = pool.submit(disconnect, barrier)
                surviving = pool.submit(complete, 'survivor', fixture('survivor', 100), barrier)
                results.update(disconnect=cancelled.result(), survivor=surviving.result())
            results['subsequent'] = complete('subsequent', fixture('subsequent', 8), stream=True)
            (output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(results), flush=True)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


if __name__ == '__main__':
    main()
