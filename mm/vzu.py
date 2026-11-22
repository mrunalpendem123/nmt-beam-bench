# List functions that use ymm/zmm registers but contain no vzeroupper (objdump -d -C output on stdin).
import re, sys
cur, ymm, vzu, rets = None, False, False, 0
rows = []
def flush():
    if cur and ymm and not vzu and rets:
        rows.append(cur)
for line in sys.stdin:
    m = re.match(r'^[0-9a-f]+ <(.*)>:$', line)
    if m:
        flush(); cur, ymm, vzu, rets = m.group(1), False, False, 0; continue
    if '%ymm' in line or '%zmm' in line: ymm = True
    if 'vzeroupper' in line: vzu = True
    if re.search(r'\tret', line): rets += 1
flush()
pat = re.compile(sys.argv[1] if len(sys.argv) > 1 else '.')
for r in rows:
    if pat.search(r): print(r[:160])
