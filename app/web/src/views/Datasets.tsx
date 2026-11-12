import { useEffect, useState } from 'react'
import { api, sceneImageUrl } from '../api'
import { ImageTile } from '../components'
import type { DatasetCollection, Scene } from '../types'

export default function Datasets({
  scenes, collections, onChanged,
}: {
  scenes: Scene[]
  collections: DatasetCollection[]
  onChanged: () => void
}) {
  const [browse, setBrowse] = useState<string | null>(null)
  const [browseIds, setBrowseIds] = useState<string[]>([])
  const [browseTotal, setBrowseTotal] = useState(0)
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

  /* Sample ids for the opened dataset. Fetched on demand and capped: a
     generated set can hold thousands, and the point of this list is to let
     you eyeball a few, not to render all of them. */
  useEffect(() => {
    if (!browse) { setBrowseIds([]); setBrowseTotal(0); return }
    let live = true
    api.datasetScenes(browse, 60)
      .then((r) => {
        if (!live) return
        setBrowseIds(r.scene_ids)
        setBrowseTotal(r.total)
      })
      .catch(() => { if (live) { setBrowseIds([]); setBrowseTotal(0) } })
    return () => { live = false }
  }, [browse])

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

  /* Only pairs uploaded through the form above: those sit directly in
     data/custom/<name>/, so their id has no further path. A sample of a
     dataset folder is also kind "custom", and offering to delete one here
     would mean offering to delete one image pair out of someone's dataset
     from a list that claims to show uploads. */
  const custom = scenes.filter(
    (s) => s.kind === 'custom' && !s.name.includes('/'),
  )

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
            <h3 style={{ margin: 0 }}>Dataset folders ({collections.length})</h3>
            <button className="ghost small" onClick={onChanged}>rescan</button>
          </div>
          <div className="hint" style={{ marginBottom: 10 }}>
            Copy a folder of stereo pairs into
            <span className="mono"> data/custom/</span> and press
            <b> rescan</b>. Any shape works: one pair, a flat folder of
            samples, or splits with samples under them — a "pair" is just a
            folder holding <span className="mono">im0</span>/
            <span className="mono">im1</span> (or
            <span className="mono"> left</span>/
            <span className="mono">right</span>) images, optionally with
            <span className="mono"> disp0GT.pfm</span> and
            <span className="mono"> calib.txt</span> beside them. Whole
            folders are selectable in the <b>Runs</b> tab.
          </div>
          {collections.length === 0 && (
            <div className="muted" style={{ fontSize: 12.5 }}>
              No dataset folders yet.
            </div>
          )}
          <div className="table-wrap" style={{ maxHeight: 320 }}>
            <table>
              <thead>
                <tr>
                  <th>dataset</th><th>kind</th><th>pairs</th><th>subsets</th>
                  <th>GT</th><th>ndisp</th><th></th>
                </tr>
              </thead>
              <tbody>
                {collections.map((c) => {
                  const depth = collections.filter(
                    (o) => o.id !== c.id && c.id.startsWith(o.id + '/'),
                  ).length
                  const splits = Object.entries(c.splits)
                  return (
                    <tr key={c.id}>
                      <td style={{ paddingLeft: 8 + depth * 16 }} title={c.path}>
                        {c.name}
                      </td>
                      <td>{c.kind}</td>
                      <td className="num">{c.count}</td>
                      <td className="muted" style={{ fontSize: 11.5 }}>
                        {splits.length
                          ? splits.map(([k, v]) => `${k} (${v})`).join(', ')
                          : '—'}
                      </td>
                      <td className="num">{c.has_gt ? 'yes' : '—'}</td>
                      <td className="num">{c.ndisp ?? '—'}</td>
                      <td>
                        <button
                          className="ghost small"
                          onClick={() => setBrowse(browse === c.id ? null : c.id)}
                        >
                          {browse === c.id ? 'close' : 'browse'}
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          {browse && (
            <div style={{ marginTop: 10 }}>
              <div className="row-between" style={{ marginBottom: 6 }}>
                <b style={{ fontSize: 12.5 }}>{browse}</b>
                <span className="muted" style={{ fontSize: 11.5 }}>
                  showing {browseIds.length} of {browseTotal}
                </span>
              </div>
              <div className="scroll-list pick-list" style={{ maxHeight: 170 }}>
                {browseIds.map((id) => (
                  <div
                    key={id}
                    className={`pick ${preview === id ? 'on' : ''}`}
                    onClick={() => setPreview(id)}
                  >
                    <span className="mono" style={{ fontSize: 11.5 }}>{id}</span>
                  </div>
                ))}
              </div>
              <div className="hint" style={{ marginTop: 6 }}>
                Click a pair to preview it. Every pair in the set is reachable
                from the Playground's dataset browser, and the whole set is one
                checkbox in the Runs tab.
              </div>
            </div>
          )}
        </div>

        <div className="card">
          <div className="row-between" style={{ marginBottom: 10 }}>
            <h3 style={{ margin: 0 }}>Uploaded pairs ({custom.length})</h3>
            {/* Folders can also be dropped into data/custom/ by hand, so the
                catalogue needs a way to be re-read without a page reload. */}
            <button className="ghost small" onClick={onChanged}>rescan</button>
          </div>
          {custom.length === 0 && (
            <div className="muted" style={{ fontSize: 12.5 }}>
              Nothing uploaded yet. Pairs added with the form on the left live
              in <span className="mono">data/custom/&lt;name&gt;/</span> and show
              up everywhere the built-in scenes do. A folder holding many pairs
              is listed above as a dataset instead.
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
          <h3>Individually listed scenes ({scenes.length})</h3>
          <div className="hint" style={{ marginBottom: 8 }}>
            Every built-in scene plus the first {12} samples of each dataset
            folder — enough to pick one by hand in the Playground. The rest of
            a folder is reached by browsing the dataset, or run as a whole in
            the Runs tab.
          </div>
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
