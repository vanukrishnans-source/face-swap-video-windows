"""Compare two video_pipeline --dump folders (reference vs Windows app). numpy only.

usage: python parity_compare.py REF_DIR APP_DIR [label] [--json out.json]
Checks: frame selection, detections, track ids/frames, raw + smoothed landmarks, pairing (+flip), and per-frame
PSNR of the swapped output frames (whole frame and inside the region the swap changed).
"""
import json, sys, os
import numpy as np


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float('inf') if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    ref, app = args[0], args[1]; label = args[2] if len(args) > 2 else os.path.basename(app.rstrip('/'))
    jout = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
    R = {'label': label, 'checks': {}, 'frames': []}
    ok = True
    mr, ma = json.load(open(f'{ref}/meta.json')), json.load(open(f'{app}/meta.json'))
    lines = [f'== {label}']
    def chk(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond); R['checks'][name] = bool(cond)
        lines.append(f'  {name:<34} {"OK" if cond else "MISMATCH"} {detail}')
    chk('output size', (mr['W'], mr['H']) == (ma['W'], ma['H']), f"{ma['W']}x{ma['H']}")
    chk('frame selection', mr['sel'] == ma['sel'], f"{len(ma['sel'])} frames")
    dr, da = json.load(open(f'{ref}/dets.json')), json.load(open(f'{app}/dets.json'))
    chk('detections per frame', [len(x) for x in dr] == [len(x) for x in da])
    md = 0.0
    for x, y in zip(dr, da):
        for p, q in zip(x, y): md = max(md, float(np.abs(np.array(p) - np.array(q)).max()))
    R['det_max_px'] = md
    chk('detected landmarks', md < 1e-3, f'max |d| = {md:.2e} px')
    tr, ta = json.load(open(f'{ref}/tracks.json')), json.load(open(f'{app}/tracks.json'))
    same_ids = len(tr['raw']) == len(ta['raw']) and all(set(a) == set(b) for a, b in zip(tr['raw'], ta['raw']))
    chk('track ids / frames', same_ids, f"ref {[len(t) for t in tr['raw']]} app {[len(t) for t in ta['raw']]}")
    for kind in ('raw', 'smooth'):
        m = 0.0
        if same_ids:
            for a, b in zip(tr[kind], ta[kind]):
                for f in a: m = max(m, float(np.abs(np.array(a[f]) - np.array(b[f])).max()))
        R[f'{kind}_max_px'] = m
        chk(f'{kind} track points', same_ids and m < 1e-3, f'max |d| = {m:.2e} px')
    chk('pairing', (mr['assign'], mr['pairing_frame']) == (ma['assign'], ma['pairing_frame']),
        f"ref {mr['assign']}@{mr['pairing_frame']} app {ma['assign']}@{ma['pairing_frame']}")
    ks = sorted(int(f[5:-12]) for f in os.listdir(ref) if f.startswith('frame') and f.endswith('_out_rgb.npy'))
    for k in ks:
        ri, ro = np.load(f'{ref}/frame{k}_in_rgb.npy'), np.load(f'{ref}/frame{k}_out_rgb.npy')
        p_app = f'{app}/frame{k}_out_rgb.npy'
        if not os.path.exists(p_app):
            chk(f'frame {k} present', False); continue
        ai, ao = np.load(f'{app}/frame{k}_in_rgb.npy'), np.load(p_app)
        din = int(np.abs(ri.astype(int) - ai.astype(int)).max())
        changed = np.any(ro != ri, axis=2)
        pf, pr = psnr(ro, ao), (psnr(ro[changed], ao[changed]) if changed.any() else float('inf'))
        mx = int(np.abs(ro.astype(int) - ao.astype(int)).max())
        R['frames'].append(dict(k=k, decoded_max_diff=din, psnr_frame=pf, psnr_swapped_region=pr, max_diff=mx,
                                changed_px=int(changed.sum())))
        chk(f'frame {k}', din == 0 and pf > 50, f'decoded input max|d|={din}  output PSNR {pf:.1f} dB (swapped region {pr:.1f} dB, {int(changed.sum())} px) max|d|={mx}')
    R['ok'] = ok
    lines.append('  RESULT: ' + ('ALL PARITY CHECKS PASSED' if ok else 'SOME CHECKS FAILED'))
    print('\n'.join(lines))
    if jout: json.dump(R, open(jout, 'w'), indent=1)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
