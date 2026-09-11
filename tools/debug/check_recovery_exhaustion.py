"""Probe request-local exhaustion on an isolated diagnostic server; executes no tools.

The token cap deliberately targets a captured repeated-call boundary, not a
maximum acceptable reasoning length. A normal length finish is NOT a loop failure.
"""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request

from jsonschema import validate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture', type=Path, required=True)
    ap.add_argument('--probe', type=Path, required=True)
    ap.add_argument('--tokens', type=int, required=True)
    ap.add_argument('--stream', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--port', type=int, default=19002)
    ap.add_argument('--temperature', type=float, default=2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def request(body):
        return urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})

    body = json.loads(args.fixture.read_text())
    body.pop('max_tokens', None)
    body.update(model='loop-numerics', max_completion_tokens=args.tokens, stream=args.stream,
                enable_thinking=True, temperature=args.temperature, seed=args.seed)
    error = None
    events = []
    published_calls = []
    try:
        with urllib.request.urlopen(request(body), timeout=240) as response:
            if args.stream:
                for line in response:
                    if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                        continue
                    event = json.loads(line[6:])
                    events.append(event)
                    if 'error' in event:
                        error = event['error']
                    for choice in event.get('choices', []):
                        published_calls.extend(choice.get('delta', {}).get('tool_calls', []))
            else:
                events.append(json.load(response))
    except urllib.error.HTTPError as failure:
        payload = json.load(failure)
        events.append(dict(http_status=failure.code, payload=payload))
        error = payload.get('error')
    (args.output / 'response.json').write_text(json.dumps(events, indent=2) + '\n')

    probe = json.loads(args.probe.read_text())
    probe.update(model='loop-numerics', max_completion_tokens=16384, stream=False,
                 enable_thinking=True, temperature=2, seed=42)
    with urllib.request.urlopen(request(probe), timeout=180) as response:
        result = json.load(response)
    (args.output / 'probe.json').write_text(json.dumps(result, indent=2) + '\n')
    choice = result['choices'][0]
    assert choice['finish_reason'] == 'tool_calls', choice
    declarations = {t['function']['name']: t['function']['parameters'] for t in probe['tools']}
    for call in choice['message']['tool_calls']:
        function = call['function']
        validate(json.loads(function['arguments']), declarations[function['name']])
    assert not published_calls, 'rejected call escaped into the stream'
    assert error and error.get('code') == 'generation_recovery_exhausted', (
        'boundary did not establish recovery exhaustion; inspect saved response', error)
    print('explicit exhaustion; no rejected call published; subsequent declared call valid', flush=True)


if __name__ == '__main__':
    main()
