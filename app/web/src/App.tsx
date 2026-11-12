import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import type { Algorithm, DatasetCollection, Scene } from './types'
import Playground from './views/Playground'
import Compare from './views/Compare'
import Runs from './views/Runs'
import Datasets from './views/Datasets'
import Synthesize from './views/Synthesize'

type Tab = 'playground' | 'compare' | 'runs' | 'datasets' | 'synthesize'

const TABS: [Tab, string, string][] = [
  ['playground', 'Playground', 'one algorithm, live parameters'],
  ['compare', 'Compare', 'several algorithms, one scene'],
  ['runs', 'Runs', 'stored benchmarks and new batch runs'],
  ['datasets', 'Datasets', 'add image pairs and whole dataset folders'],
  ['synthesize', 'Synthesize', 'generate training images with exact ground truth'],
]

export default function App() {
  const [tab, setTab] = useState<Tab>('playground')
  const [scenes, setScenes] = useState<Scene[]>([])
  const [collections, setCollections] = useState<DatasetCollection[]>([])
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([])
  const [machine, setMachine] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  /* Scenes and datasets are rescanned together: a folder dropped into
     data/custom/ adds both a dataset and the first few of its pairs, so
     refreshing one without the other would leave the two lists disagreeing
     about what exists. */
  const loadScenes = useCallback(() => {
    api.scenes().then(setScenes).catch((e) => setError(String(e)))
    api.datasets().then(setCollections).catch((e) => setError(String(e)))
  }, [])

  useEffect(() => {
    Promise.all([api.scenes(), api.algorithms(), api.health(), api.datasets()])
      .then(([s, a, h, d]) => {
        setScenes(s)
        setAlgorithms(a)
        setMachine(h.machine)
        setCollections(d)
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false))
  }, [])

  const nAvail = algorithms.filter((a) => a.available).length
  const nPairs = collections.reduce(
    // Only top-level sets, so a train/val/test split is not counted twice.
    (n, c) => n + (collections.some(
      (o) => o.id !== c.id && c.id.startsWith(o.id + '/'),
    ) ? 0 : c.count),
    0,
  )
  const gpu = machine?.gpu as string | undefined

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          Stereo Playground
          <small>
            {nAvail}/{algorithms.length} algorithms · {collections.length}{' '}
            datasets · {nPairs} pairs
            {machine && ` · ${gpu ?? 'CPU only'}`}
          </small>
        </div>
        <nav className="tabs">
          {TABS.map(([id, label, hint]) => (
            <button
              key={id}
              className={`tab ${tab === id ? 'active' : ''}`}
              onClick={() => setTab(id)}
              title={hint}
            >
              {label}
            </button>
          ))}
        </nav>
        <span className="spacer" />
        {machine && (
          <span className="muted" style={{ fontSize: 11.5 }}>
            {String(machine.cpu_brand ?? machine.processor ?? '')}
          </span>
        )}
      </header>

      <main className="body">
        {loading && (
          <div className="banner busy">
            <span className="spin" /> loading the catalogue…
          </div>
        )}
        {error && (
          <div className="banner error">
            {error}
            <div style={{ marginTop: 6 }} className="muted">
              Is the backend running? Start it with
              <span className="mono"> python -m app.server</span>.
            </div>
          </div>
        )}
        {!loading && !error && (
          <>
            {tab === 'playground' && (
              <Playground
                scenes={scenes} algorithms={algorithms} collections={collections}
              />
            )}
            {tab === 'compare' && (
              <Compare
                scenes={scenes} algorithms={algorithms} collections={collections}
              />
            )}
            {tab === 'runs' && (
              <Runs
                scenes={scenes} algorithms={algorithms} collections={collections}
              />
            )}
            {tab === 'datasets' && (
              <Datasets
                scenes={scenes} collections={collections} onChanged={loadScenes}
              />
            )}
            {tab === 'synthesize' && <Synthesize onChanged={loadScenes} />}
          </>
        )}
      </main>
    </div>
  )
}
