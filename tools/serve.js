// A static file server with HTTP range support.
//
// Range support is the whole point. The weights are 988 MB and the browser
// needs arbitrary slices of them -- one embedding row here, one layer's
// projections there -- exactly as the Node loader reads them through a file
// descriptor. Without ranges the page would have to download the entire model
// before it could multiply anything.
//
//   node tools/serve.js [port]

import { createReadStream, statSync, existsSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const port = Number(process.argv[2] ?? 8099);

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wgsl': 'text/plain; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.safetensors': 'application/octet-stream',
  '.bin': 'application/octet-stream',
};

const server = createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const requested = decodeURIComponent(url.pathname);
  const relative = normalize(requested).replace(/^([/\\])+/, '');
  const path = join(ROOT, relative === '' ? 'web/index.html' : relative);

  // Refuse anything that escapes the repository. The server exists to hand the
  // browser this project's files and nothing else.
  if (!path.startsWith(ROOT + sep) && path !== ROOT) {
    res.writeHead(403).end('outside the repository');
    return;
  }
  if (!existsSync(path) || statSync(path).isDirectory()) {
    res.writeHead(404).end(`not found: ${relative}`);
    return;
  }

  const size = statSync(path).size;
  const type = TYPES[extname(path)] ?? 'application/octet-stream';
  const headers = {
    'content-type': type,
    'accept-ranges': 'bytes',
    // The harness compares CPU and GPU in one tab and reads timings, so it
    // wants a high-resolution clock and SharedArrayBuffer available.
    'cross-origin-opener-policy': 'same-origin',
    'cross-origin-embedder-policy': 'require-corp',
    'cache-control': 'no-cache',
  };

  const range = req.headers.range;
  if (!range) {
    res.writeHead(200, { ...headers, 'content-length': size });
    createReadStream(path).pipe(res);
    return;
  }

  const match = /^bytes=(\d*)-(\d*)$/.exec(range.trim());
  if (!match) {
    res.writeHead(416, { 'content-range': `bytes */${size}` }).end();
    return;
  }
  // A range may be open at either end: "bytes=100-" or "bytes=-500", the
  // second meaning the last 500 bytes rather than everything up to 500.
  let start;
  let end;
  if (match[1] === '') {
    const suffix = Number(match[2]);
    start = Math.max(0, size - suffix);
    end = size - 1;
  } else {
    start = Number(match[1]);
    end = match[2] === '' ? size - 1 : Math.min(Number(match[2]), size - 1);
  }
  if (!Number.isFinite(start) || !Number.isFinite(end) || start > end || start >= size) {
    res.writeHead(416, { 'content-range': `bytes */${size}` }).end();
    return;
  }

  res.writeHead(206, {
    ...headers,
    'content-length': end - start + 1,
    'content-range': `bytes ${start}-${end}/${size}`,
  });
  createReadStream(path, { start, end }).pipe(res);
});

server.listen(port, '127.0.0.1', () => {
  console.log(`serving ${ROOT} at http://127.0.0.1:${port}/`);
  console.log(`  probe:   http://127.0.0.1:${port}/web/probe.html`);
  console.log(`  harness: http://127.0.0.1:${port}/web/index.html`);
});
