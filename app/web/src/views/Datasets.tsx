import { useState } from 'react'
import { api, sceneImageUrl } from '../api'
import { ImageTile } from '../components'
import type { Scene } from '../types'

export default function Datasets({
  scenes, onChanged,
}: { scenes: Scene[]; onChanged: () => void }) {
  const [name, setName] = useState('')
  const [ndisp, setNdisp] = useState(128)
  const [left, setLeft] = useState<File | null>(null)
  const [right, setRight] = useState<File | null>(null)
  const [gt, setGt] = useState<File | null>(null)
  const [seed, setSeed] = useState(42)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [note, setNote] = useState<string | null>(null)
  const [preview, setPreview] = useState<string | null>(null)

  const upload = async () => {
    if (!left || !right) { setError('pick both a left and a right image'); return }
    setBusy(true); setError(null); setNote(null)
    try {
      const form = new FormData()
      form.set('name', name)
      form.set('ndisp', String(ndisp))
      form.set('left', left)
      form.set('right', right)
      if (gt) form.set('ground_truth', gt)
      const scene = await api.addDataset(form)
      setNote(`added ${scene.id} — ${scene.width}x${scene.height}`)
      setName(''); setLeft(null); setRight(null); setGt(null)
      onChanged()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const remove = async (scene: Scene) => {
    setBusy(true); setError(null); setNote(null)
    try {
      await api.deleteDataset(scene.name)
      setNote(`removed ${scene.id}`)
      if (preview === scene.id) setPreview(null)
      onChanged()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const sample = async () => {
    setBusy(true); setError(null)
    try {
      const scene = await api.synthetic(seed)
      setNote(`generated ${scene.id} with exact ground truth`)
      setPreview(scene.id)
      onChanged()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const custom = scenes.filter((s) => s.kind === 'custom')

  return (
    <div className="split">
      <div>
        <div className="card">
          <h3>Add an image pair</h3>
          <label className="field">
            <span className="lbl"><b>name</b></span>
            <input
              type="text" value={name} placeholder="my-rig-01"
              onChange={(e) => setName(e.target.value)}
            />
            <div className="hint">Letters, digits, space, dot, dash, underscore.</div>
          </label>
          <label className="field">
            <span className="lbl">
              <b>disparity range</b><span className="val">{ndisp}</span>
            </span>
            <input
              type="range" min={16} max={256} step={16} value={ndisp}
              onChange={(e) => setNdisp(Number(e.target.value))}
            />
            <div className="hint">
              How far to search. Too small and far objects are missed; too large
              and every matcher gets slower in proportion.
            </div>
          </label>
          <label className="field">
            <span className="lbl"><b>left image</b></span>
            <input type="file" accept="image/*"
                   onChange={(e) => setLeft(e.target.files?.[0] ?? null)} />
          </label>
          <label className="field">
            <span className="lbl"><b>right image</b></span>
            <input type="file" accept="image/*"
                   onChange={(e) => setRight(e.target.files?.[0] ?? null)} />
          </label>
          <label className="field">
            <span className="lbl"><b>ground truth</b> <span className="muted">optional</span></span>
            <input type="file" accept=".pfm,image/*"
                   onChange={(e) => setGt(e.target.files?.[0] ?? null)} />
            <div className="hint">
              A float <span className="mono">.pfm</span> disparity map, or an
              image where 0 means unknown. Without it you still get timing and
              the disparity map, just no accuracy numbers.
            </div>
          </label>
          <button className="primary" style={{ width: '100%' }}
                  onClick={() => void upload()} disabled={busy || !name || !left || !right}>
            {busy ? 'uploading…' : 'add dataset'}
          </button>
          <div className="hint" style={{ marginTop: 8 }}>
            The pair must already be <b>rectified</b>: matching only searches
            along a row, so any vertical offset between the views is
            unrecoverable here.
          </div>
        </div>

        <div className="card">
          <h3>Generate a sample</h3>
          <label className="field">
            <span className="lbl">
              <b>seed</b><span className="val">{seed}</span>
            </span>
            <input
              type="range" min={0} max={200} step={1} value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
          </label>
          <button style={{ width: '100%' }} onClick={() => void sample()} disabled={busy}>
            generate synthetic pair
          </button>
          <div className="hint" style={{ marginTop: 8 }}>
            Textured scene warped by a known disparity field, so the ground
            truth is exact by construction rather than measured. Useful as a
            sanity floor — a matcher that struggles here has a bug, not a hard
            scene.
          </div>
        </div>
      </div>

      <div>
        {error && <div className="banner error" style={{ marginBottom: 12 }}>{error}</div>}
        {note && <div className="banner" style={{ marginBottom: 12 }}>{note}</div>}

        <div className="card">
          <div className="row-between" style={{ marginBottom: 10 }}>
            <h3 style={{ margin: 0 }}>Your datasets ({custom.length})</h3>
            {/* Folders can also be dropped into data/custom/ by hand, so the
                catalogue needs a way to be re-read without a page reload. */}
            <button className="ghost small" onClick={onChanged}>rescan</button>
          </div>
          {custom.length === 0 && (
            <div className="muted" style={{ fontSize: 12.5 }}>
              Nothing uploaded yet. Added pairs live in
              <span className="mono"> data/custom/</span> and show up everywhere
              the built-in scenes do.
            </div>
          )}
          {custom.map((s) => (
            <div key={s.id} className="row-between"
                 style={{ padding: '7px 0', borderBottom: '1px solid var(--border)' }}>
              <div>
                <div><b>{s.name}</b></div>
                <div className="muted" style={{ fontSize: 11.5 }}>
                  {s.width}x{s.height} · ndisp {s.ndisp} ·{' '}
                  {s.has_gt ? 'has ground truth' : 'no ground truth'}
                </div>
              </div>
              <div className="card-row">
                <button className="small" onClick={() => setPreview(s.id)}>preview</button>
                <button className="small danger" onClick={() => void remove(s)}>
                  delete
                </button>
              </div>
            </div>
          ))}
        </div>

        <div className="card">
          <h3>All scenes ({scenes.length})</h3>
          <div className="table-wrap" style={{ maxHeight: 300 }}>
            <table>
              <thead>
                <tr>
                  <th>scene</th><th>kind</th><th>size</th><th>ndisp</th>
                  <th>GT</th><th>MP</th><th></th>
                </tr>
              </thead>
              <tbody>
                {scenes.map((s) => (
                  <tr key={s.id}>
                    <td>{s.name}</td>
                    <td>{s.kind}</td>
                    <td className="num">{s.width}x{s.height}</td>
                    <td className="num">{s.ndisp}</td>
                    <td className="num">
                      {s.has_gt ? `${((s.gt_coverage ?? 0) * 100).toFixed(0)}%` : '—'}
                    </td>
                    <td className="num">{s.megapixels?.toFixed(2)}</td>
                    <td>
                      <button className="ghost small" onClick={() => setPreview(s.id)}>
                        preview
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {preview && (
          <div className="card">
            <div className="row-between" style={{ marginBottom: 10 }}>
              <h3 style={{ margin: 0 }}>Preview &middot; {preview}</h3>
              <button className="ghost small" onClick={() => setPreview(null)}>close</button>
            </div>
            <div className="img-grid">
              <ImageTile title="left" src={sceneImageUrl(preview, 'left')} />
              <ImageTile title="right" src={sceneImageUrl(preview, 'right')} />
              {scenes.find((s) => s.id === preview)?.has_gt && (
                <>
                  <ImageTile title="ground truth" src={sceneImageUrl(preview, 'gt')} />
                  <ImageTile
                    title="non-occluded mask"
                    note="the default scoring region"
                    src={sceneImageUrl(preview, 'nonocc')}
                  />
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
